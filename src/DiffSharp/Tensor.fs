﻿namespace DiffSharp
open DiffSharp.RawTensor
open DiffSharp.Util

[<CustomEquality; CustomComparison>]
type Tensor = 
    | Tensor of RawTensor
    | TensorF of Tensor * Tensor * uint32
    | TensorR of Tensor * (Tensor ref) * TensorOp * (uint32 ref) * uint32

    member t.Primal =
        match t with
        | Tensor(_) -> t
        | TensorF(tp,_,_) -> tp
        | TensorR(tp,_,_,_,_) -> tp

    member t.PrimalRaw =
        let rec primalRaw x =
            match x with
            | Tensor(tp) -> tp
            | TensorF(tp,_,_) -> primalRaw tp
            | TensorR(tp,_,_,_,_) -> primalRaw tp
        primalRaw t

    member t.Derivative
        with get() =
            match t with
            | Tensor(_) -> failwith "Cannot get derivative of constant Tensor"
            | TensorF(_,td,_) -> td
            | TensorR(_,td,_,_,_) -> !td
        and set(value) =
            match t with
            | Tensor(_) -> failwith "Cannot set derivative of constant Tensor"
            | TensorF(_) -> failwith "Cannot set derivative of TensorF"
            | TensorR(_,td,_,_,_) -> td := value

    member t.Fanout
        with get() =
            match t with
            | Tensor(_) -> failwith "Cannot get fanout of constant Tensor"
            | TensorF(_) -> failwith "Cannot get fanout of TensorF"
            | TensorR(_,_,_,f,_) -> !f
        and set(value) =
            match t with
            | Tensor(_) -> failwith "Cannot set fanout of constant Tensor"
            | TensorF(_) -> failwith "Cannot set fanout of TensorF"
            | TensorR(_,_,_,f,_) -> f := value

    // member inline t.Value = 0.
    member t.GetForward(derivative:Tensor, ?tag:uint32) = 
        let tag = defaultArg tag GlobalTagger.Next
        if t.Shape = derivative.Shape then TensorF(t, derivative, tag) else invalidArg "derivative" (sprintf "Expecting derivative of same shape with primal. primal: %A, derivative: %A" t derivative)
    member t.GetReverse(?tag:uint32) = 
        let tag = defaultArg tag GlobalTagger.Next
        TensorR(t, ref (t.Zero()), NewT, ref 0u, tag)
    member t.Shape = t.PrimalRaw.Shape
    member t.Dim = t.PrimalRaw.Dim
    member t.ToArray() = t.PrimalRaw.ToArray()
    member t.ToValue() = t.PrimalRaw.ToValue()
    member t.Zero() = Tensor(t.PrimalRaw.Zero())
    member t.Create(value) = Tensor(t.PrimalRaw.Create(value))
    override t.Equals(other) =
        match other with
        | :? Tensor as tensor -> t.PrimalRaw.Equals(tensor.PrimalRaw)
        | _ -> false
    member t.ApproximatelyEqual(tensor:Tensor, ?tolerance) =
        let tolerance = defaultArg tolerance 0.01
        t.PrimalRaw.ApproximatelyEquals(tensor.PrimalRaw, tolerance)
    override t.GetHashCode() =
        match t with
        | Tensor(tp) -> hash (tp)
        | TensorF(tp,td,tt) -> hash (tp, td, tt)
        | TensorR(tp,td,_,_,tt) -> hash (tp, !td, tt)
    interface System.IComparable with
        override t.CompareTo(other) =
            match other with
            | :? Tensor as tensor -> 
                if t.Dim = tensor.Dim && t.Dim = 0 then
                    t.PrimalRaw.CompareTo(tensor.PrimalRaw)
                else
                    invalidOp "Cannot compare non-scalar Tensors"
            | _ -> invalidOp "Cannot compare Tensor with another type"
    static member inline op_Explicit(tensor:Tensor):'a = downcast tensor.PrimalRaw.ToValue()
    static member ZerosLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.Zeros(tensor.Shape))
    static member OnesLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.Ones(tensor.Shape))
    static member RandomLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.Random(tensor.Shape))
    static member RandomNormalLike(tensor:Tensor) = Tensor(tensor.PrimalRaw.RandomNormal(tensor.Shape))
    static member Zeros(shape:int[], ?dtype:DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend CPUBase
        match dtype, device, backend with
        | Float32, CPU, CPUBase -> Tensor(RawTensorFloat32CPUBase.Zeros(shape))
        | _ -> failwithf "Unsupported Tensor creation with dtype: %A, device: %A, backend: %A" dtype device backend
    static member Ones(shape:int[], ?dtype:DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend CPUBase
        match dtype, device, backend with
        | Float32, CPU, CPUBase -> Tensor(RawTensorFloat32CPUBase.Ones(shape))
        | _ -> failwithf "Unsupported Tensor creation with dtype: %A, device: %A, backend: %A" dtype device backend
    static member Random(shape:int[], ?dtype:DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend CPUBase
        match dtype, device, backend with
        | Float32, CPU, CPUBase -> Tensor(RawTensorFloat32CPUBase.Random(shape))
        | _ -> failwithf "Unsupported Tensor creation with dtype: %A, device: %A, backend: %A" dtype device backend
    static member RandomNormal(shape:int[], ?dtype:DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend CPUBase
        match dtype, device, backend with
        | Float32, CPU, CPUBase -> Tensor(RawTensorFloat32CPUBase.RandomNormal(shape))
        | _ -> failwithf "Unsupported Tensor creation with dtype: %A, device: %A, backend: %A" dtype device backend
    static member Create(value:obj, ?dtype:DType, ?device:Device, ?backend:Backend) =
        let dtype = defaultArg dtype Float32
        let device = defaultArg device CPU
        let backend = defaultArg backend CPUBase
        match dtype, device, backend with
        | Float32, CPU, CPUBase -> Tensor(RawTensorFloat32CPUBase.Create(value))
        | _ -> failwithf "Unsupported Tensor creation with dtype: %A, device: %A, backend: %A" dtype device backend

    static member Extend(a:Tensor, shape:int[]) =
        if a.Dim <> 0 then invalidArg "tensor" (sprintf "Expecting a 0d Tensor, received shape: %A" a.Shape)
        match a with
        | Tensor(ap) -> Tensor(ap.Extend(shape))
        | TensorF(ap,ad,at) ->
            let cp = Tensor.Extend(ap, shape)
            let cd = Tensor.Extend(ad, shape)
            TensorF(cp,cd,at)
        | TensorR(ap,_,_,_,at) ->
            let cp = Tensor.Extend(ap, shape)
            TensorR(cp, ref (a.Zero()), MakeTofT0(a), ref 0u, at)

    static member Stack(tensors:seq<Tensor>) = 
        // TODO: check if all Tensors are of the same type (Tensor, TensorF, or TensorR) and have the same nesting tag
        match Seq.head tensors with
        | Tensor(ap) -> Tensor(ap.StackTs(tensors |> Seq.map (fun t -> t.PrimalRaw)))
        | TensorF(_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.Primal)
            let ad = tensors |> Seq.map (fun t -> t.Derivative)
            TensorF(Tensor.Stack(ap), Tensor.Stack(ad), at)
        | TensorR(_,_,_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.Primal)
            let cp = Tensor.Stack(ap) in TensorR(cp, ref (cp.Zero()), StackTs(tensors), ref 0u, at)

    static member Unstack (a:Tensor) =
        match a with
        | Tensor(ap) -> ap.UnstackT() |> Seq.map Tensor
        | TensorF(ap,ad,at) -> Seq.map2 (fun p d -> TensorF(p,d,at)) (ap.Unstack()) (ad.Unstack())
        | TensorR(ap,_,_,_,at) -> Seq.mapi (fun i p -> TensorR(p, ref (p.Zero()), UnstackT(a, i), ref 0u, at)) (ap.Unstack())
    member t.Unstack() = Tensor.Unstack(t)

    static member inline OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev) =
        match a with
        | Tensor(ap)           -> Tensor(fRaw(ap))
        | TensorF(ap,ad,at)    -> let cp = fTensor(ap) in TensorF(cp, dfTensorFwd(cp,ap,ad), at)
        | TensorR(ap,_,_,_,at) -> let cp = fTensor(ap) in TensorR(cp, ref (a.Zero()), dfTensorRev(a), ref 0u, at)

    static member inline OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT) =
        match a, b with
        | Tensor(ap),           Tensor(bp)                      -> Tensor(fRaw(ap, bp))
        | Tensor(_),            TensorF(bp,bd,bt)               -> let cp = fTensor(a,bp)  in TensorF(cp, dfTensorFwdCT(cp,bp,bd), bt)
        | Tensor(_),            TensorR(bp,_,_,_,bt)            -> let cp = fTensor(a,bp)  in TensorR(cp, ref (a.Zero()), dfTensorRevCT(a,b), ref 0u, bt)
        | TensorF(ap,ad,at),    Tensor(_)                       -> let cp = fTensor(ap,b)  in TensorF(cp, dfTensorFwdTC(cp,ap,ad), at)
        | TensorF(ap,ad,at),    TensorF(bp,bd,bt)    when at=bt -> let cp = fTensor(ap,bp) in TensorF(cp, dfTensorFwdTT(cp,ap,ad,bp,bd), at)
        | TensorF(ap,ad,at),    TensorF(_,_,bt)      when at>bt -> let cp = fTensor(ap,b)  in TensorF(cp, dfTensorFwdTC(cp,ap,ad), at)
        | TensorF(_,_,at),      TensorF(bp,bd,bt)    when at<bt -> let cp = fTensor(a,bp)  in TensorF(cp, dfTensorFwdCT(cp,bp,bd), bt)
        | TensorF(_,_,at),      TensorR(_,_,_,_,bt)  when at=bt -> failwith "Cannot have TensorF and TensorR in the same nesting level"
        | TensorF(ap,ad,at),    TensorR(_,_,_,_,bt)  when at>bt -> let cp = fTensor(ap,b)  in TensorF(cp, dfTensorFwdTC(cp,ap,ad), at)
        | TensorF(_,_,at),      TensorR(bp,_,_,_,bt) when at<bt -> let cp = fTensor(a,bp)  in TensorR(cp, ref (a.Zero()), dfTensorRevCT(a,b), ref 0u, bt)
        | TensorR(ap,_,_,_,at), Tensor(_)                       -> let cp = fTensor(ap,b)  in TensorR(cp, ref (a.Zero()), dfTensorRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorF(_,_,bt)      when at=bt -> failwith "Cannot have TensorR and TensorF in the same nesting level"
        | TensorR(ap,_,_,_,at), TensorF(_,_,bt)      when at>bt -> let cp = fTensor(ap, b) in TensorR(cp, ref (a.Zero()), dfTensorRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorF(bp,bd,bt)    when at<bt -> let cp = fTensor(a,bp)  in TensorF(cp, dfTensorFwdCT(cp, bp, bd), bt)
        | TensorR(ap,_,_,_,at), TensorR(bp,_,_,_,bt) when at=bt -> let cp = fTensor(ap,bp) in TensorR(cp, ref (a.Zero()), dfTensorRevTT(a,b), ref 0u, at)
        | TensorR(ap,_,_,_,at), TensorR(_,_,_,_,bt)  when at>bt -> let cp = fTensor(ap,b)  in TensorR(cp, ref (a.Zero()), dfTensorRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorR(bp,_,_,_,bt) when at<bt -> let cp = fTensor(a,bp)  in TensorR(cp, ref (a.Zero()), dfTensorRevCT(a,b), ref 0u, bt)
        | _ -> failwith "Unexpected combination of Tensors" // Won't happen, added for suppressing "incomplete matches" warning

    static member (+) (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.AddTT(b)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = bd
            let inline dfTensorRevTT(a,b) = AddTT(a,b)
            let inline dfTensorRevTC(a,b) = AddTTConst(a)
            let inline dfTensorRevCT(a,b) = AddTTConst(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a,b:RawTensor) = b.AddTT0(a)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = Tensor.Extend(ad, b.Shape)
            let inline dfTensorFwdCT(cp,bp,bd) = bd
            let inline dfTensorRevTT(a,b) = AddTT0(b,a)
            let inline dfTensorRevTC(a,b) = AddTConstT0(a)
            let inline dfTensorRevCT(a,b) = AddTT0Const(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.AddTT0(b)
            let inline fTensor(a,b) = a + b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = Tensor.Extend(bd, a.Shape)
            let inline dfTensorRevTT(a,b) = AddTT0(a,b)
            let inline dfTensorRevTC(a,b) = AddTT0Const(a)
            let inline dfTensorRevCT(a,b) = AddTConstT0(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 2 && b.Dim = 1 then
            if a.Shape.[0] = b.Shape.[0] then
                let inline fRaw(a:RawTensor,b) = a.AddT2T1(b)
                let inline fTensor(a,b) = a + b
                let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
                let inline dfTensorFwdTC(cp,ap,ad) = ad
                let inline dfTensorFwdCT(cp,bp,bd) = Tensor.ZerosLike(cp) + bd
                let inline dfTensorRevTT(a,b) = AddT2T1(a,b)
                let inline dfTensorRevTC(a,b) = AddT2T1Const(a)
                let inline dfTensorRevCT(a,b) = AddT2ConstT1(b)
                Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
            else invalidOp <| sprintf "Cannot add Tensors with shapes %A, %A" a.Shape b.Shape                
        elif a.Dim = 1 && b.Dim = 2 then
            if a.Shape.[0] = b.Shape.[0] then
                let inline fRaw(a,b:RawTensor) = b.AddT2T1(a)
                let inline fTensor(a,b) = a + b
                let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad + bd
                let inline dfTensorFwdTC(cp,ap,ad) = ad + Tensor.ZerosLike(cp)
                let inline dfTensorFwdCT(cp,bp,bd) = bd
                let inline dfTensorRevTT(a,b) = AddT2T1(b,a)
                let inline dfTensorRevTC(a,b) = AddT2ConstT1(a)
                let inline dfTensorRevCT(a,b) = AddT2T1Const(b)
                Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
            else invalidOp <| sprintf "Cannot add Tensors with shapes %A, %A" a.Shape b.Shape                
        // TODO: implement general broadcasting additions
        else failwithf "Cannot add Tensors with shapes %A, %A" a.Shape b.Shape
    static member (+) (a:Tensor, b) = a + a.Create(b)
    static member (+) (a, b:Tensor) = b.Create(a) + b

    static member (-) (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.SubTT(b)
            let inline fTensor(a,b) = a - b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = -bd
            let inline dfTensorRevTT(a,b) = SubTT(a,b)
            let inline dfTensorRevTC(a,b) = SubTTConst(a)
            let inline dfTensorRevCT(a,b) = SubTConstT(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.SubT0T(b)
            let inline fTensor(a,b) = a - b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let inline dfTensorFwdTC(cp,ap,ad) = Tensor.Extend(ad, b.Shape)
            let inline dfTensorFwdCT(cp,bp,bd) = -bd
            let inline dfTensorRevTT(a,b) = SubT0T(a,b)
            let inline dfTensorRevTC(a,b) = SubT0TConst(a)
            let inline dfTensorRevCT(a,b) = SubT0ConstT(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.SubTT0(b)
            let inline fTensor(a,b) = a - b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = ad - bd
            let inline dfTensorFwdTC(cp,ap,ad) = ad
            let inline dfTensorFwdCT(cp,bp,bd) = Tensor.Extend(-bd, a.Shape)
            let inline dfTensorRevTT(a,b) = SubTT0(a,b)
            let inline dfTensorRevTC(a,b) = SubTT0Const(a)
            let inline dfTensorRevCT(a,b) = SubTConstT0(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else failwithf "Cannot subtract Tensors with shapes %A, %A" a.Shape b.Shape
    static member (-) (a:Tensor, b) = a - a.Create(b)
    static member (-) (a, b:Tensor) = b.Create(a) - b

    static member (*) (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.MulTT(b)
            let inline fTensor(a,b) = a * b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = (ad * bp) + (ap * bd)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * b
            let inline dfTensorFwdCT(cp,bp,bd) = a * bd
            let inline dfTensorRevTT(a,b) = MulTT(a,b)
            let inline dfTensorRevTC(a,b) = MulTTConst(a,b)
            let inline dfTensorRevCT(a,b) = MulTTConst(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a,b:RawTensor) = b.MulTT0(a)
            let inline fTensor(a,b) = a * b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = (ad * bp) + (ap * bd)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * b
            let inline dfTensorFwdCT(cp,bp,bd) = a * bd
            let inline dfTensorRevTT(a,b) = MulTT0(b,a)
            let inline dfTensorRevTC(a,b) = MulTConstT0(a,b)
            let inline dfTensorRevCT(a,b) = MulTT0Const(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.MulTT0(b)
            let inline fTensor(a,b) = a * b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = (ad * bp) + (ap * bd)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * b
            let inline dfTensorFwdCT(cp,bp,bd) = a * bd
            let inline dfTensorRevTT(a,b) = MulTT0(a,b)
            let inline dfTensorRevTC(a,b) = MulTT0Const(a,b)
            let inline dfTensorRevCT(a,b) = MulTConstT0(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        // TODO: implement general broadcasting?
        else failwithf "Cannot add Tensors with shapes %A, %A" a.Shape b.Shape
    static member (*) (a:Tensor, b) = a * a.Create(b)
    static member (*) (a, b:Tensor) = b.Create(a) * b

    static member (/) (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.DivTT(b)
            let inline fTensor(a,b) = a / b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = (ad - bd * cp) / bp
            let inline dfTensorFwdTC(cp,ap,ad) = ad / b
            let inline dfTensorFwdCT(cp,bp,bd) = -bd * cp / bp
            let inline dfTensorRevTT(a,b) = DivTT(a,b)
            let inline dfTensorRevTC(a,b) = DivTTConst(a,b)
            let inline dfTensorRevCT(a,b) = DivTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.DivT0T(b)
            let inline fTensor(a,b) = a / b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = (ad - bd * cp) / bp
            let inline dfTensorFwdTC(cp,ap,ad) = ad / b
            let inline dfTensorFwdCT(cp,bp,bd) = -bd * cp / bp
            let inline dfTensorRevTT(a,b) = DivT0T(a,b)
            let inline dfTensorRevTC(a,b) = DivT0TConst(a,b)
            let inline dfTensorRevCT(a,b) = DivT0ConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.DivTT0(b)
            let inline fTensor(a,b) = a / b
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = (ad - bd * cp) / bp
            let inline dfTensorFwdTC(cp,ap,ad) = ad / b
            let inline dfTensorFwdCT(cp,bp,bd) = -bd * cp / bp
            let inline dfTensorRevTT(a,b) = DivTT0(a,b)
            let inline dfTensorRevTC(a,b) = DivTT0Const(a,b)
            let inline dfTensorRevCT(a,b) = DivTConstT0(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else failwithf "Cannot divide Tensors with shapes %A, %A" a.Shape b.Shape
    static member (/) (a:Tensor, b) = a / a.Create(b)
    static member (/) (a, b:Tensor) = b.Create(a) / b

    static member Pow (a:Tensor, b:Tensor) =
        if a.Shape = b.Shape then
            let inline fRaw(a:RawTensor,b) = a.PowTT(b)
            let inline fTensor(a,b) = a ** b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp,bd) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let inline dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let inline dfTensorRevTT(a,b) = PowTT(a,b)
            let inline dfTensorRevTC(a,b) = PowTTConst(a,b)
            let inline dfTensorRevCT(a,b) = PowTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif a.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.PowT0T(b)
            let inline fTensor(a,b) = a ** b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp,bd) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let inline dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let inline dfTensorRevTT(a,b) = PowT0T(a,b)
            let inline dfTensorRevTC(a,b) = PowT0TConst(a,b)
            let inline dfTensorRevCT(a,b) = PowT0ConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        elif b.Dim = 0 then
            let inline fRaw(a:RawTensor,b) = a.PowTT0(b)
            let inline fTensor(a,b) = a ** b
            let inline dfTensorFwdTT(cp:Tensor,ap:Tensor,ad:Tensor,bp,bd) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfTensorFwdTC(cp,ap,ad) = ad * (ap ** (b - 1.)) * b
            let inline dfTensorFwdCT(cp,bp,bd) = bd * cp * log a
            let inline dfTensorRevTT(a,b) = PowTT0(a,b)
            let inline dfTensorRevTC(a,b) = PowTT0Const(a,b)
            let inline dfTensorRevCT(a,b) = PowTConstT0(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else failwithf "Cannot exponentiate Tensors with shapes %A, %A" a.Shape b.Shape
    static member Pow (a:Tensor, b) = a ** a.Create(b)
    static member Pow (a, b:Tensor) = b.Create(a) ** b

    static member MatMul (a:Tensor, b:Tensor) =
        if a.Dim <> 2 || b.Dim <> 2 then invalidOp <| sprintf "Expecting two 2d Tensors, received Tensors with shapes %A, %A" a.Shape b.Shape
        if a.Shape.[1] = b.Shape.[0] then
            let inline fRaw(a:RawTensor,b) = a.MatMulT2T2(b)
            let inline fTensor(a,b) = Tensor.MatMul(a, b)
            let inline dfTensorFwdTT(cp,ap,ad,bp,bd) = Tensor.MatMul(ad, bp) + Tensor.MatMul(ap, bd)
            let inline dfTensorFwdTC(cp,ap,ad) = Tensor.MatMul(ad, b)
            let inline dfTensorFwdCT(cp,bp,bd) = Tensor.MatMul(a, bd)
            let inline dfTensorRevTT(a,b) = MatMulT2T2(a,b)
            let inline dfTensorRevTC(a,b) = MatMulT2T2Const(a,b)
            let inline dfTensorRevCT(a,b) = MatMulT2ConstT2(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfTensorFwdTT, dfTensorFwdTC, dfTensorFwdCT, dfTensorRevTT, dfTensorRevTC, dfTensorRevCT)
        else failwithf "Cannot multiply Tensors with shapes %A, %A" a.Shape b.Shape

    static member (~-) (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.NegT()
        let inline fTensor(a) = -a
        let inline dfTensorFwd(cp,ap,ad) = -ad
        let inline dfTensorRev(a) = NegT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Neg() = -t

    static member Sum (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SumT()
        let inline fTensor(a) = Tensor.Sum(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.Sum(ad)
        let inline dfTensorRev(a) = SumT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sum() = Tensor.Sum(t)

    static member SumT2Dim0 (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SumT2Dim0()
        let inline fTensor(a) = Tensor.SumT2Dim0(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.SumT2Dim0(ad)
        let inline dfTensorRev(a) = SumT2Dim0(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.SumT2Dim0() = Tensor.SumT2Dim0(t)
    
    static member Transpose (a:Tensor) =
        if a.Dim <> 2 then invalidOp <| sprintf "Expecting a 2d Tensor, received Tensor with shape %A" a.Shape
        let inline fRaw(a:RawTensor) = a.TransposeT2()
        let inline fTensor(a) = Tensor.Transpose(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.Transpose(ad)
        let inline dfTensorRev(a) = TransposeT2(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Transpose() = Tensor.Transpose(t)

    static member Sign (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SignT()
        let inline fTensor(a) = Tensor.Sign(a)
        let inline dfTensorFwd(cp,ap,ad) = Tensor.ZerosLike(cp)
        let inline dfTensorRev(a) = SignT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sign() = Tensor.Sign(t)

    static member Abs (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.AbsT()
        let inline fTensor(a) = Tensor.Abs(a)
        let inline dfTensorFwd(cp,ap,ad) = ad * Tensor.Sign(ap)
        let inline dfTensorRev(a) = AbsT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Abs() = Tensor.Abs(t)

    static member ReLU (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.ReLUT()
        let inline fTensor(a) = Tensor.ReLU(a)
        let inline dfTensorFwd(cp,ap,ad) = let sap = Tensor.Sign(ap) in ad * Tensor.Abs(sap) * (1. + sap) / 2.
        let inline dfTensorRev(a) = ReLUT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.ReLU() = Tensor.ReLU(t)

    static member Exp (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.ExpT()
        let inline fTensor(a) = Tensor.Exp(a)
        let inline dfTensorFwd(cp,ap,ad) = ad * cp
        let inline dfTensorRev(a) = ExpT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Exp() = Tensor.Exp(t)

    static member Log (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.LogT()
        let inline fTensor(a) = Tensor.Log(a)
        let inline dfTensorFwd(cp,ap,ad) = ad / ap
        let inline dfTensorRev(a) = LogT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Log() = Tensor.Log(t)

    static member Sqrt (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.SqrtT()
        let inline fTensor(a) = Tensor.Sqrt(a)
        let inline dfTensorFwd(cp:Tensor,ap,ad) = ad / (2. * cp)
        let inline dfTensorRev(a) = SqrtT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfTensorFwd, dfTensorRev)
    member t.Sqrt() = Tensor.Sqrt(t)

    member t.Reverse(?value:Tensor) =
        let value = defaultArg value (Tensor.OnesLike(t))
        if value.Shape <> t.Shape then invalidArg "value" <| sprintf "Expecting an adjoint value of shape %A, but received of shape %A" t.Shape value.Shape
        t.ReverseReset()
        t.ReversePush(value)

    member inline t.Backward(value) = t.Reverse(value)

    member t.ReverseReset() =
        let rec reset (ts: Tensor list) =
            match ts with
            | [] -> ()
            | t :: tt ->
                match t with
                | TensorR(_,_,o,_,_) ->
                    t.Derivative <- t.Zero()
                    t.Fanout <- t.Fanout + 1u
                    if t.Fanout = 1u then
                        match o with
                        | AddTT(a,b) -> reset (a::b::tt)
                        | AddTTConst(a) -> reset (a::tt)
                        | AddTT0(a,b) -> reset (a::b::tt)
                        | AddTT0Const(a) -> reset (a::tt)
                        | AddTConstT0(b) -> reset (b::tt)
                        | AddT2T1(a,b) -> reset (a::b::tt)
                        | AddT2T1Const(a) -> reset (a::tt)
                        | AddT2ConstT1(b) -> reset (b::tt)
                        | SubTT(a,b) -> reset (a::b::tt)
                        | SubTTConst(a) -> reset (a::tt)
                        | SubTConstT(b) -> reset (b::tt)
                        | SubT0T(a,b) -> reset (a::b::tt)
                        | SubT0TConst(a) -> reset (a::tt)
                        | SubT0ConstT(b) -> reset (b::tt)
                        | SubTT0(a,b) -> reset (a::b::tt)
                        | SubTT0Const(a) -> reset (a::tt)
                        | SubTConstT0(b) -> reset (b::tt)
                        | MulTT(a,b) -> reset (a::b::tt)
                        | MulTTConst(a,_) -> reset (a::tt)
                        | MulTT0(a,b) -> reset (a::b::tt)
                        | MulTConstT0(_,b) -> reset (b::tt)
                        | MulTT0Const(a,_) -> reset (a::tt)
                        | DivTT(a,b) -> reset (a::b::tt)
                        | DivTTConst(a,_) -> reset (a::tt)
                        | DivTConstT(_,b) -> reset (b::tt)
                        | DivT0T(a,b) -> reset (a::b::tt)
                        | DivT0TConst(a,_) -> reset (a::tt)
                        | DivT0ConstT(_,b) -> reset (b::tt)
                        | DivTT0(a,b) -> reset (a::b::tt)
                        | DivTT0Const(a,_) -> reset (a::tt)
                        | DivTConstT0(_,b) -> reset (b::tt)
                        | PowTT(a,b) -> reset (a::b::tt)
                        | PowTTConst(a,_) -> reset (a::tt)
                        | PowTConstT(_,b) -> reset (b::tt)
                        | PowT0T(a,b) -> reset (a::b::tt)
                        | PowT0TConst(a,_) -> reset (a::tt)
                        | PowT0ConstT(_,b) -> reset (b::tt)
                        | PowTT0(a,b) -> reset (a::b::tt)
                        | PowTT0Const(a,_) -> reset (a::tt)
                        | PowTConstT0(_,b) -> reset (b::tt)
                        | MatMulT2T2(a,b) -> reset (a::b::tt)
                        | MatMulT2T2Const(a,_) -> reset (a::tt)
                        | MatMulT2ConstT2(_,b) -> reset (b::tt)
                        | NegT(a) -> reset (a::tt)
                        | SumT(a) -> reset (a::tt)
                        | SumT2Dim0(a) -> reset (a::tt)
                        | MakeTofT0(a) -> reset (a::tt)
                        | StackTs(a) -> reset (List.append (a |> List.ofSeq) tt)
                        | UnstackT(a,_) -> reset (a::tt)
                        | TransposeT2(a) -> reset (a::tt)
                        | SignT(a) -> reset (a::tt)
                        | AbsT(a) -> reset (a::tt)
                        | ReLUT(a) -> reset (a::tt)
                        | ExpT(a) -> reset (a::tt)
                        | LogT(a) -> reset (a::tt)
                        | SqrtT(a) -> reset (a::tt)
                        | NewT -> reset tt
                    else reset tt
                | _ -> reset tt
        reset [t]

    member t.ReversePush(value:Tensor) =
        let rec push (ts:(Tensor*Tensor) list) =
            match ts with
            | [] -> ()
            | (v, t) :: tt ->
                match t with
                | TensorR(_,_,o,_,_) ->
                    t.Derivative <- t.Derivative + v
                    t.Fanout <- t.Fanout - 1u
                    if t.Fanout = 0u then
                        match o with
                        | AddTT(a,b) -> push ((t.Derivative, a) :: (t.Derivative, b) :: tt)
                        | AddTTConst(a) -> push ((t.Derivative, a) :: tt)
                        | AddTT0(a,b) -> push ((t.Derivative, a) :: (t.Derivative.Sum(), b) :: tt)
                        | AddTT0Const(a) -> push ((t.Derivative, a) :: tt)
                        | AddTConstT0(b) -> push ((t.Derivative.Sum(), b) :: tt)
                        | AddT2T1(a,b) -> push ((t.Derivative, a) :: (t.Derivative.SumT2Dim0(), b) :: tt)
                        | AddT2T1Const(a) -> push ((t.Derivative, a) :: tt)
                        | AddT2ConstT1(b) -> push ((t.Derivative.SumT2Dim0(), b) :: tt)
                        | SubTT(a,b) -> push ((t.Derivative, a) :: (-t.Derivative, b) :: tt)
                        | SubTTConst(a) -> push ((t.Derivative, a) :: tt)
                        | SubTConstT(b) -> push ((-t.Derivative, b) :: tt)
                        | SubT0T(a,b) -> push ((t.Derivative.Sum(), a) :: (-t.Derivative, b) :: tt)
                        | SubT0TConst(a) -> push ((t.Derivative.Sum(), a) :: tt)
                        | SubT0ConstT(b) -> push ((-t.Derivative, b) :: tt)
                        | SubTT0(a,b) -> push ((t.Derivative, a) :: (-t.Derivative.Sum(), b) :: tt)
                        | SubTT0Const(a) -> push ((t.Derivative, a) :: tt)
                        | SubTConstT0(b) -> push ((-t.Derivative.Sum(), b) :: tt)      
                        | MulTT(a,b) -> push ((t.Derivative * b.Primal, a) :: (t.Derivative * a.Primal, b) :: tt)
                        | MulTTConst(a,b) -> push ((t.Derivative * b, a) :: tt)
                        | MulTT0(a,b) -> push ((t.Derivative * b.Primal, a) :: ((t.Derivative * a.Primal).Sum(), b) :: tt)
                        | MulTConstT0(a,b) -> push (((t.Derivative * a).Sum(), b) :: tt)
                        | MulTT0Const(a,b) -> push ((t.Derivative * b, a) :: tt)
                        | DivTT(a,b) -> push ((t.Derivative / b.Primal, a) :: ((t.Derivative * (-a.Primal / (b.Primal * b.Primal))), b) :: tt)
                        | DivTTConst(a,b) -> push ((t.Derivative / b, a) :: tt)
                        | DivTConstT(a,b) -> push (((t.Derivative * (-a / (b.Primal * b.Primal))), b) :: tt)
                        | DivT0T(a,b) -> push (((t.Derivative / b.Primal).Sum(), a) :: ((t.Derivative * (-a.Primal / (b.Primal * b.Primal))), b) :: tt)
                        | DivT0TConst(a,b) -> push (((t.Derivative / b).Sum(), a) :: tt)
                        | DivT0ConstT(a,b) -> push (((t.Derivative * (-a / (b.Primal * b.Primal))), b) :: tt)
                        | DivTT0(a,b) -> push ((t.Derivative / b.Primal, a) :: ((t.Derivative * (-a.Primal / (b.Primal * b.Primal))).Sum(), b) :: tt)
                        | DivTT0Const(a,b) -> push ((t.Derivative / b, a) :: tt)
                        | DivTConstT0(a,b) -> push (((t.Derivative * (-a / (b.Primal * b.Primal))).Sum(), b) :: tt)
                        | PowTT(a,b) -> push ((t.Derivative * (a.Primal ** (b.Primal - 1.)) * b.Primal, a) :: (t.Derivative * (a.Primal ** b.Primal) * log a.Primal, b) :: tt)
                        | PowTTConst(a,b) -> push ((t.Derivative * (a.Primal ** (b - 1.)) * b, a) :: tt)
                        | PowTConstT(a,b) -> push ((t.Derivative * (a ** b.Primal) * log a, b) :: tt)
                        | PowT0T(a,b) -> push (((t.Derivative * (a.Primal ** (b.Primal - 1.)) * b.Primal).Sum(), a) :: (t.Derivative * (a.Primal ** b.Primal) * log a.Primal, b) :: tt)
                        | PowT0TConst(a,b) -> push (((t.Derivative * (a.Primal ** (b - 1.)) * b).Sum(), a) :: tt)
                        | PowT0ConstT(a,b) -> push ((t.Derivative * (a ** b.Primal) * log a, b) :: tt)
                        | PowTT0(a,b) -> push ((t.Derivative * (a.Primal ** (b.Primal - 1.)) * b.Primal, a) :: ((t.Derivative * (a.Primal ** b.Primal) * log a.Primal).Sum(), b) :: tt)
                        | PowTT0Const(a,b) -> push ((t.Derivative * (a.Primal ** (b - 1.)) * b, a) :: tt)
                        | PowTConstT0(a,b) -> push (((t.Derivative * (a ** b.Primal) * log a).Sum(), b) :: tt)
                        | MatMulT2T2(a,b) -> push ((Tensor.MatMul(t.Derivative, b.Primal.Transpose()), a) :: (Tensor.MatMul(a.Primal.Transpose(), t.Derivative), b) :: tt)
                        | MatMulT2T2Const(a,b) -> push ((Tensor.MatMul(t.Derivative, b.Transpose()), a) :: tt)
                        | MatMulT2ConstT2(a,b) -> push ((Tensor.MatMul(a.Transpose(), t.Derivative), b) :: tt)
                        | NegT(a) -> push ((-t.Derivative, a) :: tt)
                        | SumT(a) -> push ((Tensor.Extend(t.Derivative, a.Shape), a) :: tt)
                        | SumT2Dim0(a) -> push ((Tensor.ZerosLike(a) + t.Derivative, a) :: tt)
                        | MakeTofT0(a) -> push ((t.Derivative.Sum(), a) :: tt)
                        | StackTs(a) ->  push (List.append (a |> Seq.map2 (fun t a -> (t, a)) (t.Derivative.Unstack()) |> Seq.toList) tt)
                        | UnstackT(a,i) -> failwith "ReversePush UnstackT not implemented"
                        | TransposeT2(a) -> push ((t.Derivative.Transpose(), a) :: tt)
                        | SignT(a) -> push ((Tensor.ZerosLike(a), a) :: tt)
                        | AbsT(a) -> push ((t.Derivative * a.Primal.Sign(), a) :: tt)
                        | ReLUT(a) -> let sap = a.Primal.Sign() in push ((t.Derivative * (sap.Abs()) * (sap + 1.) / 2., a) :: tt)
                        | ExpT(a) -> push ((t.Derivative * t.Primal, a) :: tt)
                        | LogT(a) -> push ((t.Derivative / a.Primal, a) :: tt)
                        | SqrtT(a) -> push ((t.Derivative / (2. * t.Primal), a) :: tt)
                        | NewT -> push tt
                    else push tt
                | _ -> push tt
        push [(value, t)]

and TensorOp =
    | AddTT of Tensor * Tensor
    | AddTTConst of Tensor
    | AddTT0 of Tensor * Tensor
    | AddTT0Const of Tensor
    | AddTConstT0 of Tensor
    | AddT2T1 of Tensor * Tensor
    | AddT2T1Const of Tensor
    | AddT2ConstT1 of Tensor
    
    | SubTT of Tensor * Tensor
    | SubTTConst of Tensor
    | SubTConstT of Tensor
    | SubT0T of Tensor * Tensor
    | SubT0TConst of Tensor
    | SubT0ConstT of Tensor
    | SubTT0 of Tensor * Tensor
    | SubTT0Const of Tensor
    | SubTConstT0 of Tensor

    | MulTT of Tensor * Tensor
    | MulTTConst of Tensor * Tensor
    | MulTT0 of Tensor * Tensor
    | MulTConstT0 of Tensor * Tensor
    | MulTT0Const of Tensor * Tensor

    | DivTT of Tensor * Tensor
    | DivTTConst of Tensor * Tensor
    | DivTConstT of Tensor * Tensor
    | DivT0T of Tensor * Tensor
    | DivT0TConst of Tensor * Tensor
    | DivT0ConstT of Tensor * Tensor
    | DivTT0 of Tensor * Tensor
    | DivTT0Const of Tensor * Tensor
    | DivTConstT0 of Tensor * Tensor

    | PowTT of Tensor * Tensor
    | PowTTConst of Tensor * Tensor
    | PowTConstT of Tensor * Tensor
    | PowT0T of Tensor * Tensor
    | PowT0TConst of Tensor * Tensor
    | PowT0ConstT of Tensor * Tensor
    | PowTT0 of Tensor * Tensor
    | PowTT0Const of Tensor * Tensor
    | PowTConstT0 of Tensor * Tensor

    | MatMulT2T2 of Tensor * Tensor
    | MatMulT2T2Const of Tensor * Tensor
    | MatMulT2ConstT2 of Tensor * Tensor

    | NegT of Tensor
    | SumT of Tensor
    | SumT2Dim0 of Tensor
    | MakeTofT0 of Tensor
    | StackTs of seq<Tensor>
    | UnstackT of Tensor * int
    | TransposeT2 of Tensor
    | SignT of Tensor
    | AbsT of Tensor
    | ReLUT of Tensor
    | ExpT of Tensor
    | LogT of Tensor
    | SqrtT of Tensor
    | NewT

[<RequireQualifiedAccess>]
[<CompilationRepresentation (CompilationRepresentationFlags.ModuleSuffix)>]
module Tensor =
    let create (count:int) (value:float32) = Tensor.Create(Array.create count value)
    let zeroCreate (count:int) = Tensor.Create(Array.zeroCreate count)
    let init (count:int) (initializer:int->float32) = Tensor.Create(Array.init count initializer)
    let shape (t:Tensor) = t.Shape
    let dim (t:Tensor) = t.Dim