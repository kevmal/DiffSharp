﻿//
// This file is part of
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright (C) 2014, National University of Ireland Maynooth.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with DiffSharp. If not, see <http://www.gnu.org/licenses/>.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

#light

/// Forward AD module, lazy higher-order
module DiffSharp.AD.ForwardN

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// DualN numeric type, where the tangent value is another DualN, forming a lazy chain of higher-order derivatives
// UNOPTIMIZED
[<CustomEquality; CustomComparison>]
type DualN =
    // Primal, tangent
    | DualN of float * Lazy<DualN>
    static member Create(p) = DualN(p, lazy (DualN.Zero))
    static member Create(p, t) = DualN(p, lazy (DualN.Create(t)))
    static member op_Explicit(p) = DualN.Create(p)
    static member op_Explicit(DualN(p, _)) = p
    override d.ToString() = let (DualN(p, t)) = d in sprintf "DualN (%f, %f)" p (float t.Value)
    member d.P = let (DualN(p, _)) = d in p
    member d.T = let (DualN(_, t)) = d in t.Value
    static member Zero = DualN(0., lazy (DualN.Zero))
    static member One = DualN(1., lazy (DualN.Zero))
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? DualN as d2 -> let DualN(a, _), DualN(b, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this DualN with another type of object."
    override d.Equals(other) = 
        match other with
        | :? DualN as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (DualN(a, b)) = d in hash [|a; b|]
    // DualN - DualN binary operations
    static member (+) (a:DualN, b:DualN) = DualN(a.P + b.P, lazy (a.T + b.T))
    static member (-) (a:DualN, b:DualN) = DualN(a.P - b.P, lazy (a.T - b.T))
    static member (*) (a:DualN, b:DualN) = DualN(a.P * b.P, lazy (a.T * b + a * b.T))
    static member (/) (a:DualN, b:DualN) = DualN(a.P / b.P, lazy ((a.T * b - a * b.T) / (b * b)))
    static member Pow (a:DualN, b:DualN) = DualN(a.P ** b.P, lazy ((a ** b) * ((b * a.T / a) + ((log a) * b.T))))
    static member Atan2 (a:DualN, b:DualN) = DualN(atan2 a.P b.P, lazy ((a.T * b - a * b.T) / (a * a + b * b)))
    // DualN - float binary operations
    static member (+) (a:DualN, b) = DualN(a.P + b, lazy (a.T))
    static member (-) (a:DualN, b) = DualN(a.P - b, lazy (a.T))
    static member (*) (a:DualN, b) = DualN(a.P * b, lazy (a.T * b))
    static member (/) (a:DualN, b) = DualN(a.P / b, lazy (a.T / b))
    static member Pow (a:DualN, b) = DualN(a.P ** b, lazy (b * (a ** (b - 1.)) * a.T))
    static member Atan2 (a:DualN, b) = DualN(atan2 a.P b, lazy ((b * a.T) / (b * b + a * a)))
    // float - DualN binary operations
    static member (+) (a, b:DualN) = DualN(b.P + a, lazy (b.T))
    static member (-) (a, b:DualN) = DualN(a - b.P, lazy (-b.T))
    static member (*) (a, b:DualN) = DualN(b.P * a, lazy (b.T * a))
    static member (/) (a, b:DualN) = DualN(a / b.P, lazy (-a * b.T / (b * b)))
    static member Pow (a, b:DualN) = DualN(a ** b.P, lazy ((DualN.Create(a) ** b) * (log a) * b.T))
    static member Atan2 (a, b:DualN) = DualN(atan2 a b.P, lazy (-(a * b.T) / (a * a + b * b)))
    // DualN - int binary operations
    static member (+) (a:DualN, b:int) = a + float b
    static member (-) (a:DualN, b:int) = a - float b
    static member (*) (a:DualN, b:int) = a * float b
    static member (/) (a:DualN, b:int) = a / float b
    static member Pow (a:DualN, b:int) = DualN.Pow(a, float b)
    static member Atan2 (a:DualN, b:int) = DualN.Atan2(a, float b)
    // int - DualN binary operations
    static member (+) (a:int, b:DualN) = (float a) + b
    static member (-) (a:int, b:DualN) = (float a) - b
    static member (*) (a:int, b:DualN) = (float a) * b
    static member (/) (a:int, b:DualN) = (float a) / b
    static member Pow (a:int, b:DualN) = DualN.Pow(float a, b)
    static member Atan2 (a:int, b:DualN) = DualN.Atan2(float a, b)
    // DualN unary operations
    static member Abs (a:DualN) = 
        if a.P = 0. then invalidArg "" "The derivative of abs is not defined at 0."
        DualN(abs a.P, lazy (a.T * float (sign a.P)))
    static member Log (a:DualN) = DualN(log a.P, lazy (a.T / a))
    static member Log10 (a:DualN) = DualN(log10 a.P, lazy (a.T / (a * log10val)))
    static member Exp (a:DualN) = DualN(exp a.P, lazy (a.T * exp a))
    static member Sin (a:DualN) = DualN(sin a.P, lazy (a.T * cos a))
    static member Cos (a:DualN) = DualN(cos a.P, lazy (-a.T * sin a))
    static member Tan (a:DualN) = DualN(tan a.P, lazy (a.T / ((cos a) * (cos a))))
    static member (~-) (a:DualN) = DualN(-a.P, lazy (-a.T))
    static member Sqrt (a:DualN) = DualN(sqrt a.P, lazy (a.T / (2. * sqrt a)))
    static member Sinh (a:DualN) = DualN(sinh a.P, lazy (a.T * cosh a))
    static member Cosh (a:DualN) = DualN(cosh a.P, lazy (a.T * sinh a))
    static member Tanh (a:DualN) = DualN(tanh a.P, lazy (a.T / ((cosh a) * (cosh a))))
    static member Asin (a:DualN) = DualN(asin a.P, lazy (a.T / sqrt (1. - a * a)))
    static member Acos (a:DualN) = DualN(acos a.P, lazy (-a.T / sqrt (1. - a * a)))
    static member Atan (a:DualN) = DualN(atan a.P, lazy (a.T / (1. + a * a)))

/// DualN operations module (automatically opened)
[<AutoOpen>]
module DualNOps =
    /// Make DualN, with primal value `p` and tangent 0
    let inline dualN p = DualN.Create(p, 0.)
    /// Make DualN, with primal value `p` and tangent value `t`
    let inline dualNSet (p, t) = DualN.Create(p, t)
    /// Make active DualN (i.e. variable of differentiation), with primal value `p` and tangent 1
    let inline dualNAct p = DualN.Create(p, 1.)
    /// Make an array of arrays of DualN, with primal values given in array `x`. The tangent values along the diagonal are 1, the rest are 0.
    let inline dualNActArrayArray (x:float[]) = Array.init x.Length (fun i -> (Array.init x.Length (fun j -> if i = j then dualNAct x.[j] else dualN x.[j])))
    /// Get the primal value of a DualN
    let inline primal (DualN(p, _)) = p
    /// Get the tangent value of a DualN
    let inline tangent (DualN(_, t)) = float t.Value
    /// Get the tangent-of-tangent value of a DualN
    let inline tangent2 (d:DualN) = tangent d.T
    /// Get the primal and tangent value of a DualN, as a tuple
    let inline tuple (DualN(p, t)) = (p, float t.Value)

    /// Compute the `n`-th derivative of a DualN
    let rec diffLazy n =
        match n with
        | a when a < 0 -> invalidArg "n" "Order of derivative cannot be negative."
        | 0 -> fun (x:DualN) -> x
        | 1 -> fun x -> x.T
        | _ -> fun x -> diffLazy (n - 1) x.T

    /// Custom operator (/^) for differentiation. Usage: `x` /^ `n`, value of the `n`-th order derivative of `x`.
    let ( /^ ) x n =
        diffLazy n x |> primal


/// ForwardN differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardNOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        dualNAct x |> f |> tuple
    
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        dualNAct x |> f |> tangent

    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' f x =
        dualNAct x |> f |> fun a -> (primal a, tangent2 a)
        
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 f x =
        dualNAct x |> f |> tangent2

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' f x =
        dualNAct x |> f |> fun a -> (primal a, tangent a, tangent2 a)

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn n f x =
        dualNAct x |> f |> diffLazy n |> primal

    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' n f x =
        let orig = x |> dualNAct |> f
        let d = orig |> diffLazy n
        (primal orig, primal d)

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' f x v =
        Array.zip x v |> Array.map dualNSet |> f |> tuple

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv f x v =
        gradv' f x v |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x =
        let a = Array.map f (dualNActArrayArray x)
        (primal a.[0], Array.map tangent a)

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd
            
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f x =
        let a = Array.map f (dualNActArrayArray x)
        (let (DualN(p, _)) = a.[0] in p, Array.sumBy tangent2 a)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x =
        let a = Array.map f (dualNActArrayArray x)
        (Array.map primal a.[0], Array2D.map tangent (array2D a))

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        jacobianT' f x |> fun (v, j) -> (v, transpose j)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' f x v = 
        Array.zip x v |> Array.map dualNSet |> f |> Array.map tuple |> Array.unzip

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv f x v = 
        jacobianv' f x v |> snd

/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' (f:DualN->DualN) (x:float) = ForwardNOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:DualN->DualN) (x:float) = ForwardNOps.diff f x
    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' (f:DualN->DualN) (x:float) = ForwardNOps.diff2' f x
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 (f:DualN->DualN) (x:float) = ForwardNOps.diff2 f x
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' (f:DualN->DualN) (x:float) = ForwardNOps.diff2'' f x
    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' (n:int) (f:DualN->DualN) (x:float) = ForwardNOps.diffn' n f x 
    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn (n:int) (f:DualN->DualN) (x:float) = ForwardNOps.diffn n f x
    /// Original value and directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' (f:Vector<DualN>->DualN) (x:Vector<float>) (v:Vector<float>) = ForwardNOps.gradv' (vector >> f) (array x) (array v)
    /// Directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv (f:Vector<DualN>->DualN) (x:Vector<float>) (v:Vector<float>) = ForwardNOps.gradv (vector >> f) (array x) (array v)
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<DualN>->DualN) (x:Vector<float>) = ForwardNOps.grad' (vector >> f) (array x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<DualN>->DualN) (x:Vector<float>) = ForwardNOps.grad (vector >> f) (array x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' (f:Vector<DualN>->DualN) (x:Vector<float>) = ForwardNOps.laplacian' (vector >> f) (array x)
    /// Laplacian of a vector-to-scalar function `f`, at point x
    let inline laplacian (f:Vector<DualN>->DualN) (x:Vector<float>) = ForwardNOps.laplacian (vector >> f) (array x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<DualN>->Vector<DualN>) (x:Vector<float>) = ForwardNOps.jacobianT' (vector >> f >> array) (array x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<DualN>->Vector<DualN>) (x:Vector<float>) = ForwardNOps.jacobianT (vector >> f >> array) (array x) |> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<DualN>->Vector<DualN>) (x:Vector<float>) = ForwardNOps.jacobian' (vector >> f >> array) (array x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<DualN>->Vector<DualN>) (x:Vector<float>) = ForwardNOps.jacobian (vector >> f >> array) (array x) |> Matrix.ofArray2D
    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' (f:Vector<DualN>->Vector<DualN>) (x:Vector<float>) (v:Vector<float>) = ForwardNOps.jacobianv' (vector >> f >> array) (array x) (array v) |> fun (a, b) -> (vector a, vector b)
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv (f:Vector<DualN>->Vector<DualN>) (x:Vector<float>) (v:Vector<float>) = ForwardNOps.jacobianv (vector >> f >> array) (array x) (array v) |> vector