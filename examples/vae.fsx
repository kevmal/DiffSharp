(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/netcoreapp3.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Backends.Torch.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // IPYNB

(*** condition: fsx ***)
#if FSX
// This is a workaround for https://github.com/dotnet/fsharp/issues/10136, necessary in F# scripts and .NET Interactive
System.Runtime.InteropServices.NativeLibrary.Load(let path1 = System.IO.Path.GetDirectoryName(typeof<DiffSharp.dsharp>.Assembly.Location) in if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/linux-x64/native/libtorch.so" else path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/win-x64/native/torch_cpu.dll")
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // FSX

(*** condition: ipynb ***)
#if IPYNB
// This is a workaround for https://github.com/dotnet/fsharp/issues/10136, necessary in F# scripts and .NET Interactive
System.Runtime.InteropServices.NativeLibrary.Load(let path1 = System.IO.Path.GetDirectoryName(typeof<DiffSharp.dsharp>.Assembly.Location) in if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/linux-x64/native/libtorch.so" else path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/win-x64/native/torch_cpu.dll")

// Set up formatting for notebooks
Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

open System
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data
open DiffSharp.ShapeChecking

type VAE(xDim:Int, zDim:Int, ?hDims:seq<Int>, ?activation:Tensor->Tensor, ?activationLast:Tensor->Tensor) =
    inherit Model()
    let hDims = defaultArg hDims (let d = (xDim+zDim)/2 in seq [d; d]) |> Array.ofSeq
    let activation = defaultArg activation dsharp.relu
    let activationLast = defaultArg activationLast dsharp.sigmoid
    let dims = [| yield xDim; yield! hDims; yield zDim |]
            
    let enc = Array.append [|for i in 0..dims.Length-2 -> Linear(dims.[i], dims.[i+1])|] [|Linear(dims.[dims.Length-2], dims.[dims.Length-1])|]
    let dec = [|for i in 0..dims.Length-2 -> Linear(dims.[i+1], dims.[i])|] |> Array.rev
    do 
        base.add([for m in enc -> box m])
        base.add([for m in dec -> box m])

    let encode (x: Tensor) =
        //printfn "encode: x.shapex = %A" x.shapex
        let mutable x = x
        for i in 0..enc.Length-3 do
            x <- activation <| enc.[i].forward(x)
        let mu = enc.[enc.Length-2].forward(x)
        let logVar = enc.[enc.Length-1].forward(x)
        mu, logVar

    let latent mu (logVar:Tensor) =
        let std = dsharp.exp(0.5*logVar)
        let eps = dsharp.randnLike(std)
        eps.mul(std).add(mu)

    let decode (z: Tensor) =
        //printfn "decode: z.shapex = %A" z.shapex
        let mutable h = z
        for i in 0..dec.Length-2 do
            h <- activation <| dec.[i].forward(h)
        activationLast <| dec.[dec.Length-1].forward(h)

    member _.encodeDecode(x:Tensor) =
        //printfn "encodeDecode: x.shapex = %A" x.shapex
        let mu, logVar = encode (x.viewx(Shape [|Int -1; xDim|]))
        let z = latent mu logVar
        decode z, mu, logVar

    override m.forward(x) =
        //printfn "m.forward: x.shapex = %A" x.shapex
        let x, _, _ = m.encodeDecode(x) in x

    override _.ToString() = sprintf "VAE(%A, %A, %A)" xDim hDims zDim

    static member loss(xRecon:Tensor, x:Tensor, mu:Tensor, logVar:Tensor) =
        let bce = dsharp.bceLoss(xRecon, x.view([|-1; 28*28|]), reduction="sum")
        let kl = -0.5 * dsharp.sum(1. + logVar - mu.pow(2.) - logVar.exp())
        bce + kl

    member m.loss(x: Tensor) =
        //printfn "loss: x.shapex = %A" x.shapex
        let xRecon, mu, logVar = m.encodeDecode x
        VAE.loss(xRecon, x, mu, logVar)

    member _.sample(?numSamples:Int) = 
        let numSamples = defaultArg numSamples (Int 1)
        dsharp.randn(Shape [|numSamples; zDim|]) |> decode

    new (xDim:int, zDim:int, ?hDims:seq<int>, ?activation:Tensor->Tensor, ?activationLast:Tensor->Tensor) =
        VAE(Int xDim, Int zDim, ?hDims = Option.map (Seq.map Int) hDims, ?activation=activation, ?activationLast=activationLast)

(*
dsharp.config(backend=Backend.Torch, device=Device.CPU)
dsharp.seed(0)

let trainSet = MNIST("./mnist", train=true, transform=id)
let trainLoader = trainSet.loader(batchSize=32, shuffle=true)

let model = VAE(28*28, 16, [512; 256])
printfn "%A" model

let optimizer = Adam(model, lr=dsharp.tensor(0.001))

let epochs = 2
for epoch = 0 to epochs do
    for i, x, _ in trainLoader.epoch() do
        printfn "loader: x.shapex = %A" x.shapex
        model.reverseDiff()
        let l = model.loss(x)
        l.reverse()
        optimizer.step()
        printfn "epoch: %A/%A minibatch: %A/%A loss: %A" epoch epochs i trainLoader.length (float(l))

        if i % 250 = 249 then
            printfn "Saving samples"
            let samples = model.sample(Int 64).view([-1; 1; 28; 28])
            samples.saveImage(sprintf "samples_%A_%A.png" epoch i)
*)

(*
dsharp.config(device=sym?Default, dtype=sym?Default, backend=Backend.Symbolic) // symbolic default device, symbolic default dtype
dsharp.tensor([1;1])
dsharp.zeros([sym?M;sym?N])
dsharp.zeros([sym?M;sym?N]) + dsharp.zeros([sym?M;sym?N])
dsharp.zeros([sym?M;sym?N]) + dsharp.zeros([1;1])
dsharp.zeros([1;2]) + dsharp.zeros([1;1])
dsharp.zero() + dsharp.zero()


//test case v. type checking  = "works"  = no type errors for all values of right type

//test case v. shape checking = "works"  = no shape errors for all values of right shape 

/// !VAE(N*28*28, Z, [H1; H2]).loss(B*N*28*28)
dsharp.config(backend=Backend.Symbolic, device=GPU, dtype=Dtype.Float32)
dsharp.seed(0)

[<SampleModel>]

let model = VAE(N*28*28, Z, [H1; H2]).loss(B*N*28*28) 


model.nparameters
model.parameters
model.parameters.dtype
model.parameters.shape


Internally:

1.  int   --> type Dim(value: int)
2.  int[] --> type Shape(dims: Dim[]) 

Externally (API)

  dsharp.zeros(shape:int[], ...)
  dsharp.zeros(shape:Shape, ...)   <-- added overload

  dsharp.ones(shape:Shape, ...)   <-- added overload

dsharp.ones(shape:Shape, ...)   <-- added overload


*)


module Model =
    open DiffSharp.ShapeChecking

    let checkAndPrintShapes<'T when 'T :> Model> () =
      let dflt = Backend.Default
      try
        Backend.Default <- Backend.ShapeChecking
        let syms = BackendSymbolStatics.Get().CreateSymContext()
        let ctors = typeof<'T>.GetConstructors()
        let ctor = 
            ctors 
            |> Array.tryFind (fun ctor -> 
                ctor.GetParameters() |> Array.exists (fun p -> 
                    let pt = p.ParameterType.ToString()
                    pt.Contains("DiffSharp.Int")))
            |> function 
               | None -> failwith "couldn't find a model constructor taking Int parameter"
               | Some c -> c
        let args =
            ctor.GetParameters() |> Array.map (fun p -> 
                let pty = p.ParameterType
                let pts = pty.ToString()
                if pts = "DiffSharp.Int" then
                   printfn "making symbolic for model parameter %s" p.Name
                   syms.CreateInjected<Int>(p.Name) |> box
                elif pts = "Microsoft.FSharp.Core.FSharpOption`1[DiffSharp.Int]" then 
                   printfn "making symbolic for option model parameter %s" p.Name
                   Some(syms.CreateInjected<Int>(p.Name)) |> box                   
                elif pts.StartsWith("Microsoft.FSharp.Core.FSharpOption`1[") then 
                   null // None
                elif pts = "System.Int32" then 
                   let v = 1
                   printfn "assuming sample value '%d' for model parameter %s" v p.Name
                   box v
                elif pts = "System.Double" then 
                   let v = 1.0
                   printfn "assuming sample value '%f' for model parameter %s" v p.Name
                   box v
                elif pts = "System.Boolean" then 
                   let v = true
                   printfn "assuming sample value '%b' for model parameter %s" v p.Name
                   box v
                else failwithf "unknown model parameter type %O" p.ParameterType
             )
        let model = ctor.Invoke(args) :?> Model
        let transfers =
            [ for ndim in 0 .. 5 do
                let input = dsharp.zeros(Shape [| for d in 1 .. ndim -> syms.CreateInjected<Int>("N" + string d) |])
                let res = try Ok (model.forward(input)) with  e -> Error e
                yield (input, res) ]

        printfn ""
        printfn "---------------------"
        printfn "Model shape summary for %s" typeof<'T>.FullName
        
        for (KeyValue(a,b)) in model.parametersDict.values |> Seq.toArray |> Seq.sortBy (fun (KeyValue(a,b)) -> a) do
           printfn "   %s : %O" a b.value.shapex

        // Probe the forward function for shape behaviour
        printfn "   forward(...):"
        for (input, res) in transfers do
            match res with 
            | Ok res -> 
                printfn "      %O --> %O" input.shapex res.shapex
            | Error e -> 
                printfn "      %O --> fails\n%s" input.shapex (e.ToString().Split('\r','\n') |> Array.map (fun s -> "        " + s) |> String.concat "\n")
                () 
      finally
        Backend.Default <- dflt



Model.checkAndPrintShapes<Linear>()
Model.checkAndPrintShapes<VAE>()
Model.checkAndPrintShapes<Conv1d>()
Model.checkAndPrintShapes<Conv2d>()
Model.checkAndPrintShapes<Conv3d>()
