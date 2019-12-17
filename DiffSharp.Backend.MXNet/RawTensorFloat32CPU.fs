namespace DiffSharp.Backend.None
open DiffSharp
open DiffSharp.Backend
open DiffSharp.Util
open System
open MXNetSharp

type RawTensorFloat32CPU(value: NDArray, shape:int[]) =
    inherit RawTensor(value, shape, DType.Float32, Device.CPU, DiffSharp.Backend.Backend.None)
    let context = CPU 0
    static let dtype = DataType.Float32

    member private t.IndexToFlatIndex(index:int[]) =
        let mutable flatIndex = 0
        for i=0 to index.Length - 1 do
            let v = if i = index.Length - 1 then 1 else (Array.reduce (*) t.Shape.[i+1..])
            flatIndex <- flatIndex + index.[i] * v
        flatIndex
    
    member private t.FlatIndexToIndex(flatIndex:int) =
        let index = Array.create t.Dim 0
        let mutable mul = t.Nelement
        let mutable fi = flatIndex
        for i=t.Dim downto 1 do
            mul <- mul / t.Shape.[t.Dim-i]
            index.[i-1] <- fi / mul
            fi <- fi - index.[i-1] * mul
        index |> Array.rev

    member t.Item
        with get ([<System.ParamArray>] index:int[]) =
            if index.Length <> t.Dim then invalidArg "index" (sprintf "Expecting a %id index" t.Dim)
            let tvalue = t.Value:?>NDArray
            tvalue.[index].ToFloat32Scalar()

    override t.GetItem(index:int[]) = RawTensorFloat32CPU.Create(t.[index])
    
    override t.GetSlice(bounds:int[,]) =
        let slice = [|for i = 0 to bounds.GetLength(0) - 1 do bounds.[i,0]; bounds.[i,1] |]
        let result = (t.Value:?>NDArray).GetSlice(slice)
        let shape = Array.init (bounds.GetLength(0)) (fun i -> bounds.[i,1] - bounds.[i,0] + 1) |> shapeSqueeze -1
        upcast RawTensorFloat32CPU(result, shape)

    override t1.CompareTo(t2) =
        compare (t1.ToValue():?>float32) (t2.ToValue():?>float32)
    
    override t.Create(value) = RawTensorFloat32CPU.Create(value)
    override t.Create(value, shape) =
        let value = value:?>float32
        match shape.Length with
        | 0 -> upcast RawTensorFloat32CPU(NDArray.CopyFrom([|value|],[1],context), [||])
        | _ -> upcast RawTensorFloat32CPU(context.Ones(shape,dtype).MutMultiply(double value), shape)
    override t.Zero() = upcast RawTensorFloat32CPU(NDArray.ConvertCopyFrom([|0.0|], shape = [1],ctx = context,dtype=dtype), [||])
    override t.Zeros(shape) = upcast RawTensorFloat32CPU(context.Zeros(shape, dtype), shape)
    override t.One() = upcast RawTensorFloat32CPU(NDArray.ConvertCopyFrom([|1.0|], shape = [1],ctx = context,dtype=dtype), [||])
    override t.Ones(shape) = upcast RawTensorFloat32CPU(context.Ones(shape, dtype), shape)
    override t.Random(shape) = 
        let dtype = 
            match dtype with 
            | Float32 -> FloatDType.Float32
            | Float16 -> FloatDType.Float16
            | Float64 -> FloatDType.Float64
            | x -> failwithf "Data type %A not supported with Random" x
        upcast RawTensorFloat32CPU(context.RandomUniform(0.0, 1.0, shape, dtype), shape)
    override t.RandomNormal(shape) =
        let dtype = 
            match dtype with 
            | Float32 -> FloatDType.Float32
            | Float16 -> FloatDType.Float16
            | Float64 -> FloatDType.Float64
            | x -> failwithf "Data type %A not supported with Random" x
        upcast RawTensorFloat32CPU(context.RandomNormal(0.0, 1.0, shape, dtype), shape)
    override t.RandomMultinomial(numSamples) = 
        let dtype = 
            match dtype with 
            | Float32 -> SampleMultinomialDtype.Float32
            | Float16 -> SampleMultinomialDtype.Float16
            | Float64 -> SampleMultinomialDtype.Float64
            | Int32 -> SampleMultinomialDtype.Int32
            | UInt8 -> SampleMultinomialDtype.Uint8
            | x -> failwithf "Data type %A not supported with Random" x
        let result = MX.SampleMultinomial(t.Value:?>NDArray, [numSamples], dtype=dtype)
        upcast RawTensorFloat32CPU(result.[0],[|numSamples|])

    static member RandomMultinomial(probs:RawTensor, numSamples:int):RawTensor =
        let dtype = 
            match dtype with 
            | Float32 -> SampleMultinomialDtype.Float32
            | Float16 -> SampleMultinomialDtype.Float16
            | Float64 -> SampleMultinomialDtype.Float64
            | Int32 -> SampleMultinomialDtype.Int32
            | UInt8 -> SampleMultinomialDtype.Uint8
            | x -> failwithf "Data type %A not supported with Random" x
        let result = MX.SampleMultinomial(probs.Value:?>NDArray, [numSamples], dtype=dtype)
        upcast RawTensorFloat32CPU(result.[0],[|numSamples|])

    override t.GetString() =
        // sprintf "RawTensor(Value=%A, Shape=%A, Dim=%A, Length=%A)" t.Value t.Shape t.Dim t.Length
        match t.Dim with
        | 0 -> sprintf "%A" ((t.Value:?>NDArray).ToFloat32Scalar())
        | _ ->
            let sb = System.Text.StringBuilder()
            let rec print (shape:int[]) externalCoords = 
                if shape.Length = 1 then
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        let globalCoords = Array.append externalCoords [|i|]
                        sb.Append(prefix) |> ignore
                        sb.Append(sprintf "%A" (t.[globalCoords])) |> ignore
                        prefix <- "; "
                    sb.Append("]") |> ignore
                else
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    for i=0 to shape.[0]-1 do
                        sb.Append(prefix) |> ignore
                        print shape.[1..] (Array.append externalCoords [|i|])
                        prefix <- "; "
                    sb.Append("]") |> ignore
            print t.Shape [||]
            sb.ToString()

    override t.ToValue() =
        match t.Dim with
        | 0 -> upcast (t.Value:?>NDArray).ToFloat32Scalar()
        | _ -> invalidOp (sprintf "Cannot convert %Ad Tensor to scalar" t.Dim)

    override t.ToArray() = //TODO: speed up
        match t.Dim with
        | 0 -> invalidOp "Cannot convert 0d Tensor to array"
        | 1 -> upcast Array.init t.Shape.[0] (fun i -> t.[i])
        | 2 -> upcast Array2D.init t.Shape.[0] t.Shape.[1] (fun i j -> t.[i, j])
        | 3 -> upcast Array3D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] (fun i j k -> t.[i, j, k])
        | 4 -> upcast Array4D.init t.Shape.[0] t.Shape.[1] t.Shape.[2] t.Shape.[3] (fun i j k l -> t.[i, j, k, l])
        | _ -> invalidOp (sprintf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape)

    override t1.Equals(t2:RawTensor) = 
        match t2 with
        | :? RawTensorFloat32CPU as t2 -> t1.Shape = t2.Shape && MX.Sum((t1.Value:?>NDArray) .= (t2.Value:?>NDArray)).ToFloat32Scalar() = 0.f
        | _ -> failwith <| sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override t1.ApproximatelyEquals(t2:RawTensor, tolerance) =
        let tolerance = double <| tolerance
        match t2 with
        | :? RawTensorFloat32CPU as t2 -> 
            let t1value = t1.Value:?>NDArray
            let t2value = t2.Value:?>NDArray
            t1.Shape = t2.Shape && 
                MX.Sum((t1value - t2value) .>= tolerance).ToFloat32Scalar() = 0.f
        | _ -> failwith <| sprintf "Cannot compare RawTensors of different types. t1:%A, t2:%A" t1 t2

    override __.StackTs(tensors) =
        let tensors = tensors |> Seq.toList
        let values, shapes = tensors |> List.map (fun t -> t.Value:?>NDArray, t.Shape) |> List.unzip
        if not (allEqual shapes) then invalidArg "tensors" "Expecting Tensors with same shape"
        let n = tensors |> List.length
        let m = shapeLength shapes.[0]
        let result = MX.Concat(data = (values |> List.map (fun a -> a.Reshape(1,-2)) |> List.toArray), dim = 0)
        upcast RawTensorFloat32CPU(result, Array.append [|n|] shapes.[0])

    override t.UnstackT() =
        if t.Dim < 1 then invalidOp "Cannot unstack scalar Tensor (dim < 1)"
        let tvalue = t.Value:?>NDArray
        let n = t.Shape.[0]
        let unstackedShape = if t.Dim = 1 then [||] else t.Shape |> Array.skip 1
        let unstackedLength = shapeLength unstackedShape
        Seq.init n (fun i -> tvalue.[i])
        |> Seq.map (fun v -> upcast RawTensorFloat32CPU(v, unstackedShape))

    override t1.LtTT(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = t1value .< t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.GtTT(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = t1value .> t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.LeTT(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = t1value .<= t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.GeTT(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = t1value .>= t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t.MaxIndexT() =
        let tvalue = t.Value:?>NDArray
        t.FlatIndexToIndex(MX.Argmax(tvalue.Reshape(-1)).ToIntScalar()) //REVIEW: is there an op that does this? 

    override t.MinIndexT() =
        let tvalue = t.Value:?>NDArray
        t.FlatIndexToIndex(MX.Argmin(tvalue.Reshape(-1)).ToIntScalar()) //REVIEW: is there an op that does this? 

    override t1.AddTT(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = t1value + t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddTT0(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray  //TODO: check dim?
        let result = t1value .+ t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddT2T1(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = t1value + t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.AddTTSlice(location:int[], t2) = failwith "not impl"
        (*// if not (shapeContains t1.Shape t2.Shape) then failwithf "Expecting t1.Shape to contain t2.Shape, received %A, %A" t1.Shape t2.Shape
        let t1value = t1.Value:?>NDArray
        let t2 = t2 :?> RawTensorFloat32CPU
        let result = Array.copy t1value
        let shape2 = shapeUnsqueezeAs t2.Shape t1.Shape
        let rec add (shape2:int[]) externalCoords =
            if shape2.Length = 1 then
                for i=0 to shape2.[0]-1 do
                    let globalCoords = Array.append externalCoords [|i|]
                    let t1Coords = Array.map2 (+) globalCoords location
                    let t1FlatIndex = t1.IndexToFlatIndex(t1Coords)
                    result.[t1FlatIndex] <- result.[t1FlatIndex] + t2.[globalCoords]
            else
                for i=0 to shape2.[0]-1 do
                    add (shape2.[1..]) (Array.append externalCoords [|i|])
        add shape2 [||]
        upcast RawTensorFloat32CPU(result, t1.Shape)*)

    override t1.SubTT(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = t1value - t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.SubT0T(t2) =
        let t1value = t1.Value:?>NDArray //TODO: check dim?
        let t2value = t2.Value:?>NDArray
        let result = t1value .- t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.SubTT0(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray  //TODO: check dim?
        let result = t1value .- t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MulTT(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = t1value * t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MulTT0(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray //TODO: check dim?
        let result = t2value .* t1value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.DivTT(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = t1value / t2value
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.DivT0T(t2) =
        let t1value = (t1.Value:?>NDArray) //TODO: check dim?
        let t2value = (t2.Value:?>NDArray)
        let result = t1value ./ t2value
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.DivTT0(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray //TODO: check dim?
        let result = t1value ./ t2value 
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.PowTT(t2) =
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = MX.Power(t1value, t2value)
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.PowT0T(t2) = //TODO: check dim?
        let t1value = (t1.Value:?>NDArray)
        let t2value = (t2.Value:?>NDArray)
        let result = MX.BroadcastPower(t1value, t2value)
        upcast RawTensorFloat32CPU(result, t2.Shape)

    override t1.PowTT0(t2) = //TODO: check dim?
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray
        let result = MX.BroadcastPower(t1value, t2value)
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.MatMulT2T2(t2) =
        if t1.Dim <> 2 || t2.Dim <> 2 then invalidOp <| sprintf "Expecting two 2d Tensors, received Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1rows, t1cols = t1.Shape.[0], t1.Shape.[1]
        let t2rows, t2cols = t2.Shape.[0], t2.Shape.[1]
        if t1cols <> t2rows then invalidOp <| sprintf "Cannot multiply Tensors with shapes %A, %A" t1.Shape t2.Shape
        let t1value = t1.Value:?>NDArray
        let t2value = t2.Value:?>NDArray        
        let result = MX.LinalgGemm2(t1value, t2value)
        RawTensorFloat32CPU.Create(result)
        
    override t.NegT() =
        let tvalue = t.Value:?>NDArray
        let result = -tvalue
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.SumT() =
        let tvalue = t.Value:?>NDArray
        let result = MX.Sum(tvalue)
        upcast RawTensorFloat32CPU(result, [||]) 
    
    override t.SumT2Dim0() =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let tvalue = t.Value:?>NDArray
        let result = MX.Sum(tvalue, [0], keepdims = false)
        let resultShape = [|t.Shape.[1]|]
        upcast RawTensorFloat32CPU(result, resultShape)

    override t.TransposeT2() =
        if t.Dim <> 2 then invalidOp "Expecting a 2d Tensor"
        let tvalue = t.Value:?>NDArray
        let result = tvalue.SwapAxis(0,1)
        RawTensorFloat32CPU.Create(result)

    override t.SqueezeT(dim) =
        let tvalue = t.Value:?>NDArray
        let newDim = shapeSqueeze dim t.Shape
        let result = tvalue.Reshape(newDim)
        upcast RawTensorFloat32CPU(result, newDim)

    override t.UnsqueezeT(dim) =
        let tvalue = t.Value:?>NDArray
        let newDim = shapeUnsqueeze dim t.Shape
        let result = tvalue.Reshape(newDim)
        upcast RawTensorFloat32CPU(result, newDim)

    override t.ViewT(shape:int[]) =
        if shapeLength t.Shape <> shapeLength shape then invalidOp <| sprintf "Cannot view Tensor of shape %A as shape %A" t.Shape shape
        let tvalue = t.Value:?>NDArray
        let result = tvalue.Reshape(shape)
        upcast RawTensorFloat32CPU(result, shape)

    override t.SignT() =
        let tvalue = t.Value:?>NDArray
        let result = MX.Sign(tvalue)
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.FloorT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> floor
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.CeilT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> ceil
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.RoundT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> round
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.AbsT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> abs
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t1.ReluT() =
        let t1value = t1.Value:?>NDArray
        let result = MX.Relu(t1value)
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t1.SigmoidT() =
        let t1value = t1.Value:?>NDArray
        let result = MX.Sigmoid(t1value)
        upcast RawTensorFloat32CPU(result, t1.Shape)

    override t.ExpT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> exp
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.LogT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> log
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.Log10T() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> log10
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SqrtT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> sqrt
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SinT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> sin
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.CosT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> cos
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.TanT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> tan
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.SinhT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> sinh
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.CoshT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> cosh
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.TanhT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> tanh
        upcast RawTensorFloat32CPU(result, t.Shape)

    override t.AsinT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> asin
        upcast RawTensorFloat32CPU(result, t.Shape)
        
    override t.AcosT() =
        let tvalue = t.Value:?>NDArray
        let result = tvalue |> acos
        upcast RawTensorFloat32CPU(result, t.Shape)                
        
    override t.AtanT() =
        let tvalue = t.Value:?>NDArray
        let result = atan tvalue
        upcast RawTensorFloat32CPU(result, t.Shape)

and RawTensorFloat32CPUStatics() = 

    inherit RawTensorStatics()

    static let dtype = DataType.Float32
    static let context = CPU 0

    override __.Zeros(shape:int[]):RawTensor = upcast RawTensorFloat32CPU(context.Zeros(shape, dtype), shape)
    
    override __.Ones(shape:int[]):RawTensor = upcast RawTensorFloat32CPU(context.Ones(shape, dtype), shape)

    override __.Random(shape:int[]):RawTensor =
        let dtype = 
            match dtype with 
            | Float32 -> FloatDType.Float32
            | Float16 -> FloatDType.Float16
            | Float64 -> FloatDType.Float64
            | x -> failwithf "Data type %A not supported with Random" x
        upcast RawTensorFloat32CPU(context.RandomUniform(0.0, 1.0, shape, dtype), shape)


    override __.RandomNormal(shape:int[]):RawTensor =
        let dtype = 
            match dtype with 
            | Float32 -> FloatDType.Float32
            | Float16 -> FloatDType.Float16
            | Float64 -> FloatDType.Float64
            | x -> failwithf "Data type %A not supported with Random" x
        upcast RawTensorFloat32CPU(context.RandomNormal(0.0, 1.0, shape, dtype), shape)


    override __.Create(value:obj) : RawTensor = 
        match value with 
        | :? Array as a -> 
            let result = NDArray.CopyFrom(a,context)
            upcast RawTensorFloat32CPU(result, result.Shape)
        | :? float32 as a -> 
            let result = NDArray.CopyFrom([|a|],context)
            upcast RawTensorFloat32CPU(result, [||])
        | x -> invalidArg "value" "Cannot convert value to RawTensorFloat32CPU"
    