namespace DiffSharp.ShapeChecking

open DiffSharp
open DiffSharp.Backends

[<AutoOpen>]
/// Augments the dsharp and Tensor APIs with inputs accepting potentially-symbolic shapes (Shape) and lengths/indicies (Int)
module ShapedInferenceAutoOpens =

    type Tensor with

        /// <summary>Version of Tensor.expand accepting a a possibly-symbolic shape.</summary>
        member a.expand(newShape:Shape) = a.expandx(newShape)

        /// <summary>Version of Tensor.view accepting a a possibly-symbolic shape.</summary>
        member a.view(shape:Shape) = a.viewx(shape)

    type dsharp with

        /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given shape, element type and configuration</summary>
        static member empty(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Empty(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '0' values for the given shape, element type and configuration</summary>
        static member zeros(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Zeros(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '1' values for the given shape, element type and configuration</summary>
        static member ones(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Ones(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with the scalar <paramref name="value" />, for the given shape, element type and configuration</summary>
        static member full(shape:Shape, value:obj, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Full(shape, value, ?dtype=dtype, ?device=device, ?backend=backend))

        // /// <summary>TBD</summary>
        // static member arange(endVal:int, ?startVal:int, ?step:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).arangeLike(endVal=endVal, ?startVal=startVal, ?step=step)

        // /// <summary>TBD</summary>
        // static member eye(rows:int, ?cols:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor.eye(rows=rows, ?cols=cols, ?dtype=dtype, ?device=device, ?backend=backend)

        // /// <summary>TBD</summary>
        // static member onehot(length:int, hot:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).onehotLike(length, hot)

        /// <summary>TBD</summary>
        static member rand(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Random(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>TBD</summary>
        static member randn(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.RandomNormal(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>TBD</summary>
        static member randint(low:int, high:int, shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.RandomInt(shape, low, high, ?dtype=dtype, ?device=device, ?backend=backend))

        // /// <summary>TBD</summary>
        // static member multinomial(probs:Tensor, numSamples:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = probs.multinomial(numSamples, ?dtype=dtype, ?device=device, ?backend=backend)

        // /// <summary>TBD</summary>
        // static member bernoulli(probs:Tensor, ?dtype:Dtype, ?device:Device, ?backend:Backend) = probs.bernoulli(?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>TBD</summary>
        static member zerosLike(a:Tensor, shape:Shape, ?dtype, ?device, ?backend) = a.zerosLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>TBD</summary>
        static member onesLike(a:Tensor, shape:Shape, ?dtype, ?device, ?backend) = a.onesLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>TBD</summary>
        static member fullLike(a:Tensor, value:scalar, shape:Shape, ?dtype, ?device, ?backend) = a.fullLike(value, shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        // /// <summary>TBD</summary>
        // static member arangeLike(a:Tensor, endVal:float, ?startVal:float, ?step:float, ?dtype:Dtype, ?device:Device, ?backend:Backend) = a.arangeLike(endVal=endVal, ?startVal=startVal, ?step=step, ?dtype=dtype, ?device=device, ?backend=backend)

        // /// <summary>TBD</summary>
        // static member onehotLike(a:Tensor, length:int, hot:int, ?dtype, ?device, ?backend) = a.onehotLike(length, hot, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>TBD</summary>
        static member randLike(a:Tensor, shape:Shape, ?dtype, ?device, ?backend) =
            a.randLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>TBD</summary>
        static member randnLike(a:Tensor, shape:Shape, ?dtype, ?device, ?backend) =
            a.randnLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>TBD</summary>
        static member randintLike(a:Tensor, low:int, high:int, shape:Shape, ?dtype, ?device, ?backend) =
            a.randintLike(low=low, high=high, shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>TBD</summary>
        static member fullshape(a:Tensor) = a.shapex

        // /// <summary>TBD</summary>
        //static member dilate(a:Tensor, dilations:seq<Int>) = a.dilate(dilations)

        // /// <summary>TBD</summary>
        //static member undilate(a:Tensor, dilations:seq<Int>) = a.undilate(dilations)

        // /// <summary>TBD</summary>
        //static member repeat(a:Tensor, dim:int, times:int) = a.repeat(dim, times)

        /// <summary>TBD</summary>
        static member view(a:Tensor, shape:Shape) = a.viewx(shape)

        /// <summary>TBD</summary>
        static member maxpool1d(a:Tensor, kernelSize:Int, ?stride:Int, ?padding:Int) =
            a.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding) |> fst

        /// <summary>TBD</summary>
        static member maxpool1di(a:Tensor, kernelSize:Int, ?stride:Int, ?padding:Int) =
            a.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding)

        /// <summary>TBD</summary>
        static member maxpool2d(a:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>TBD</summary>
        static member maxpool2di(a:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>TBD</summary>
        static member maxpool3d(a:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>TBD</summary>
        static member maxpool3di(a:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>TBD</summary>
        static member maxunpool1d(a:Tensor, indices:Tensor, outputSize:seq<Int>, kernelSize:Int, ?stride:Int, ?padding:Int) =
            a.maxunpool1dx(indices, kernelSize, ?stride=stride, ?padding=padding, outputSize=outputSize)

        /// <summary>TBD</summary>
        static member maxunpool2d(a:Tensor, indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxunpool2dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>TBD</summary>
        static member maxunpool3d(a:Tensor, indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxunpool3dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>TBD</summary>
        static member conv1d(a:Tensor, b:Tensor, ?stride:Int, ?padding:Int, ?dilation:int) =
            a.conv1dx(b, ?stride=stride, ?padding=padding, ?dilation=dilation)

        /// <summary>TBD</summary>
        static member conv2d(a:Tensor, b:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:int, ?dilations:seq<int>) =
            a.conv2dx(b, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>TBD</summary>
        static member conv3d(a:Tensor, b:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:int, ?dilations:seq<int>) =
            a.conv3dx(b, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>TBD</summary>
        static member convTranspose1d(a:Tensor, b:Tensor, ?stride:Int, ?padding:Int, ?dilation:int, ?outputPadding:Int) =
            a.convTranspose1dx(b, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)

        /// <summary>TBD</summary>
        static member convTranspose2d(a:Tensor, b:Tensor, ?stride:Int, ?padding:Int, ?dilation:int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<int>, ?outputPaddings:seq<Int>) = a.convTranspose2dx(b, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

        /// <summary>TBD</summary>
        static member convTranspose3d(a:Tensor, b:Tensor, ?stride:Int, ?padding:Int, ?dilation:int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<int>, ?outputPaddings:seq<Int>) = a.convTranspose3dx(b, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

        // /// <summary>TBD</summary>
        // static member pad(a:Tensor, paddings:seq<int>) = a.pad(paddings)

#if SYMBOLIC_SHAPES    
    module DeviceType =
        let Symbolic (s: ISym) : DeviceType = LanguagePrimitives.EnumOfValue (hash (s.GetVarName()))

    /// Contains functions and settings related to device specifications.
    module Device = 

        let Symbolic (sym: ISym) : Device =
            let dt = DeviceType.Symbolic(sym.SymContext.CreateVar(sym.GetVarName()))
            let device = Device(dt, 0)
            device

    let mutable private symscope = None

    /// <summary>A global symbol context for shape checking. Allows <c>sym?Name</c> notation for symbols in test code, avoiding lots of pesky strings</summary>
    let sym<'T> : ISymScope = 
        if symscope.IsNone then
            symscope <- Some (BackendSymbolStatics.Get().CreateSymContext())
        symscope.Value

    type ISymScope with 
        member syms.Device(name:string) = Device.Symbolic (syms.CreateVar(name))
        member syms.Int(name:string) = Int.FromSymbol (syms.CreateVar(name))

    /// Create a symbol in the global symbol context of the given name
    let (?) (syms: ISymScope) (name: string) : Int = syms.Int(name)

#endif
    type LiveCheckAttribute() = inherit System.Attribute()

