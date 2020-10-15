namespace DiffSharp.ShapeChecking

open DiffSharp
open DiffSharp.Backends

[<AutoOpen>]
/// Augments the dsharp and Tensor APIs with inputs accepting potentially-symbolic shapes (Shape) and lengths/indicies (Int)
module ShapedInferenceAutoOpens =

    type Tensor with

        /// <summary>Returns a new view of the object tensor with singleton dimensions expanded to a larger size.</summary>
        /// <param name="newShape">The requested shape.</param>
        member a.expand(newShape:Shape) = a.expandx(newShape)

        /// <summary>Returns a new view of the object tensor with singleton dimensions expanded to a larger size.</summary>
        /// <param name="newShape">The requested shape.</param>
        member a.expand(newShape:seq<Int>) = a.expandx(Shape newShape)

        /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.</summary>
        member a.view(shape:Shape) = a.viewx(shape)

        /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        member a.view(shape:seq<Int>) = a.viewx(Shape shape)

        // /// <summary>TBD</summary>
        //member a.dilate(dilations:seq<Int>) = a.dilate(dilations)

        // /// <summary>TBD</summary>
        //member a.undilate(dilations:seq<Int>) = a.undilate(dilations)

        // /// <summary>TBD</summary>
        //member a.repeat(dim:int, times:int) = a.repeat(dim, times)

        /// <summary>Applies a 1D max pooling over an input signal composed of several input planes.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        member a.maxpool1d(kernelSize:Int, ?stride:Int, ?padding:Int) =
            a.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding) |> fst

        /// <summary>Applies a 1D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        member a.maxpool1di(kernelSize:Int, ?stride:Int, ?padding:Int) =
            a.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding)

        /// <summary>Applies a 2D max pooling over an input signal composed of several input planes.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        member a.maxpool2d(?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>Applies a 2D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        member a.maxpool2di(?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>Applies a 3D max pooling over an input signal composed of several input planes.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        member a.maxpool3d(?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>Applies a 3D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        member a.maxpool3di(?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>Computes a partial inverse of maxpool1di</summary>
        /// <param name="indices">The indices selected by maxpool1di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        member a.maxunpool1d(indices:Tensor, outputSize:seq<Int>, kernelSize:Int, ?stride:Int, ?padding:Int) =
            a.maxunpool1dx(indices, kernelSize, ?stride=stride, ?padding=padding, outputSize=outputSize)

        /// <summary>Computes a partial inverse of maxpool2di</summary>
        /// <param name="indices">The indices selected by maxpool2di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        member a.maxunpool2d(indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxunpool2dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>Computes a partial inverse of maxpool3di</summary>
        /// <param name="indices">The indices selected by maxpool3di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        member a.maxunpool3d(indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxunpool3dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>Applies a 1D convolution over an input signal composed of several input planes</summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit paddings on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        member a.conv1d(filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:int) =
            a.conv1dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation)

        /// <summary>Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        member a.convTranspose1d(filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:int, ?outputPadding:Int) =
            a.convTranspose1dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)

        /// <summary>Applies a 2D convolution over an input signal composed of several input planes</summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        member a.conv2d(filters:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:int, ?dilations:seq<int>) =
            a.conv2dx(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>Applies a 2D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
        member a.convTranspose2d(filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<int>, ?outputPaddings:seq<Int>) = 
            a.convTranspose2dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

        /// <summary>Applies a 3D convolution over an input signal composed of several input planes</summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        member a.conv3d(filters:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:int, ?dilations:seq<int>) =
            a.conv3dx(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>Applies a 3D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
        member a.convTranspose3d(filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<int>, ?outputPaddings:seq<Int>) = a.convTranspose3dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

        // /// <summary>TBD</summary>
        // member a.pad(paddings:seq<int>) = a.pad(paddings)

    type dsharp with

        /// <summary>Returns a new view of the input tensor with singleton dimensions expanded to a larger size</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        static member expand(input:Tensor, shape:seq<Int>) = input.expand(shape)

        /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given shape, element type and configuration</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member empty(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Empty(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given shape, element type and configuration</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member empty(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Empty(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given length, element type and configuration</summary>
        /// <param name="length">The length of the returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member empty(length:Int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Empty(Shape [| length |], ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '0' values for the given shape, element type and configuration</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member zeros(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Zeros(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '0' values for the given shape, element type and configuration</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member zeros(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Zeros(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '1' values for the given shape, element type and configuration</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member ones(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Ones(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '1' values for the given shape, element type and configuration</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member ones(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Ones(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with the scalar <paramref name="value" />, for the given shape, element type and configuration</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="value">The .NET object used to form the initial values for the tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member full(shape:Shape, value:obj, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Full(shape, value, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with the scalar <paramref name="value" />, for the given shape, element type and configuration</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="value">The .NET object used to form the initial values for the tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member full(shape:seq<Int>, value:obj, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Full(Shape shape, value, ?dtype=dtype, ?device=device, ?backend=backend))

        // /// <summary>TBD</summary>
        // static member arange(endVal:int, ?startVal:int, ?step:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).arangeLike(endVal=endVal, ?startVal=startVal, ?step=step)

        // /// <summary>TBD</summary>
        // static member eye(rows:int, ?cols:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor.eye(rows=rows, ?cols=cols, ?dtype=dtype, ?device=device, ?backend=backend)

        // /// <summary>TBD</summary>
        // static member onehot(length:int, hot:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).onehotLike(length, hot)

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member rand(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Random(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member rand(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.Random(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member randn(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.RandomNormal(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member randn(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.RandomNormal(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).</summary>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member randint(low:int, high:int, shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.RandomInt(shape, low, high, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).</summary>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member randint(low:int, high:int, shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            Tensor(RawTensor.RandomInt(Shape shape, low, high, ?dtype=dtype, ?device=device, ?backend=backend))

        // /// <summary>TBD</summary>
        // static member multinomial(probs:Tensor, numSamples:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = probs.multinomial(numSamples, ?dtype=dtype, ?device=device, ?backend=backend)

        // /// <summary>TBD</summary>
        // static member bernoulli(probs:Tensor, ?dtype:Dtype, ?device:Device, ?backend:Backend) = probs.bernoulli(?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '0' values with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member zerosLike(input:Tensor, shape:Shape, ?dtype, ?device, ?backend) = input.zerosLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '0' values with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member zerosLike(input:Tensor, shape:seq<Int>, ?dtype, ?device, ?backend) = input.zerosLike(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '1' values with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member onesLike(input:Tensor, shape:Shape, ?dtype, ?device, ?backend) = input.onesLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '1' values with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member onesLike(input:Tensor, shape:seq<Int>, ?dtype, ?device, ?backend) = input.onesLike(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with the given scalar value with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="value">The scalar giving the the initial values for the tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member fullLike(input:Tensor, value:scalar, shape:Shape, ?dtype, ?device, ?backend) = input.fullLike(value, shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with the given scalar value with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="value">The scalar giving the the initial values for the tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member fullLike(input:Tensor, value:scalar, shape:seq<Int>, ?dtype, ?device, ?backend) = input.fullLike(value, shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        // /// <summary>TBD</summary>
        // static member arangeLike(input:Tensor, endVal:float, ?startVal:float, ?step:float, ?dtype:Dtype, ?device:Device, ?backend:Backend) = input.arangeLike(endVal=endVal, ?startVal=startVal, ?step=step, ?dtype=dtype, ?device=device, ?backend=backend)

        // /// <summary>TBD</summary>
        // static member onehotLike(input:Tensor, length:int, hot:int, ?dtype, ?device, ?backend) = input.onehotLike(length, hot, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1) with characteristics based on the input tensor</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randLike(input:Tensor, shape:Shape, ?dtype, ?device, ?backend) =
            input.randLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1) with characteristics based on the input tensor</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randLike(input:Tensor, shape:seq<Int>, ?dtype, ?device, ?backend) =
            input.randLike(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution) with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randnLike(input:Tensor, shape:Shape, ?dtype, ?device, ?backend) =
            input.randnLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution) with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randnLike(input:Tensor, shape:seq<Int>, ?dtype, ?device, ?backend) =
            input.randnLike(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive) with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randintLike(input:Tensor, low:int, high:int, shape:Shape, ?dtype, ?device, ?backend) =
            input.randintLike(low=low, high=high, shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive) with characteristics based on the input tensor.</summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randintLike(input:Tensor, low:int, high:int, shape:seq<Int>, ?dtype, ?device, ?backend) =
            input.randintLike(low=low, high=high, shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns the full shape information about the tensor</summary>
        static member fullshape(input:Tensor) = input.shapex

        // /// <summary>TBD</summary>
        //static member dilate(input:Tensor, dilations:seq<Int>) = input.dilate(dilations)

        // /// <summary>TBD</summary>
        //static member undilate(input:Tensor, dilations:seq<Int>) = input.undilate(dilations)

        // /// <summary>TBD</summary>
        //static member repeat(input:Tensor, dim:int, times:int) = input.repeat(dim, times)

        /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        static member view(input:Tensor, shape:Shape) = input.viewx(shape)

        /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        static member view(input:Tensor, shape:seq<Int>) = input.viewx(Shape shape)

        /// <summary>Applies a 1D max pooling over an input signal composed of several input planes.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        static member maxpool1d(input:Tensor, kernelSize:Int, ?stride:Int, ?padding:Int) =
            input.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding) |> fst

        /// <summary>Applies a 1D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        static member maxpool1di(input:Tensor, kernelSize:Int, ?stride:Int, ?padding:Int) =
            input.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding)

        /// <summary>Applies a 2D max pooling over an input signal composed of several input planes.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        static member maxpool2d(input:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>Applies a 2D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        static member maxpool2di(input:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>Applies a 3D max pooling over an input signal composed of several input planes.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        static member maxpool3d(input:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>Applies a 3D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        static member maxpool3di(input:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>Computes a partial inverse of maxpool1di</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="indices">The indices selected by maxpool1di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        static member maxunpool1d(input:Tensor, indices:Tensor, outputSize:seq<Int>, kernelSize:Int, ?stride:Int, ?padding:Int) =
            input.maxunpool1dx(indices, kernelSize, ?stride=stride, ?padding=padding, outputSize=outputSize)

        /// <summary>Computes a partial inverse of maxpool2di</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="indices">The indices selected by maxpool2di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        static member maxunpool2d(input:Tensor, indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxunpool2dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>Computes a partial inverse of maxpool3di</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="indices">The indices selected by maxpool3di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        static member maxunpool3d(input:Tensor, indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxunpool3dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>Applies a 1D convolution over an input signal composed of several input planes</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit paddings on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        static member conv1d(input:Tensor, filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:int) =
            input.conv1dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation)

        /// <summary>Applies a 2D convolution over an input signal composed of several input planes</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        static member conv2d(input:Tensor, filters:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:int, ?dilations:seq<int>) =
            input.conv2dx(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>Applies a 3D convolution over an input signal composed of several input planes</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        static member conv3d(input:Tensor, filters:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:int, ?dilations:seq<int>) =
            input.conv3dx(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        static member convTranspose1d(input:Tensor, filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:int, ?outputPadding:Int) =
            input.convTranspose1dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)

        /// <summary>Applies a 2D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
        static member convTranspose2d(input:Tensor, filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<int>, ?outputPaddings:seq<Int>) = 
            input.convTranspose2dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

        /// <summary>Applies a 3D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
        static member convTranspose3d(input:Tensor, filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<int>, ?outputPaddings:seq<Int>) = input.convTranspose3dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

        // /// <summary>TBD</summary>
        // static member pad(input:Tensor, paddings:seq<int>) = input.pad(paddings)

