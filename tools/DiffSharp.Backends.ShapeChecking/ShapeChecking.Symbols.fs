namespace rec DiffSharp.Backends.ShapeChecking

#if SYMBOLIC_SHAPES
open System
open System.Collections.Concurrent
open System.Reflection
open DiffSharp.ShapeChecking
open Microsoft.Z3

type ShapeCheckingBackendSymbolStatics() =
    inherit BackendSymbolStatics()
    override _.CreateSymContext() = SymContextImpl() :> _

[<AutoOpen>]
module internal Util = 

    let (|Mul|_|) (s:Expr) = if s.IsMul then Some (s.Args) else None
    let (|IntNum|_|) (s:Expr) = if s.IsIntNum then Some ((s :?> IntNum).Int) else None

    let rec isFreeIn (v: Expr) (tm: Expr) =
       if v = tm then true
       else if tm.IsApp then tm.Args |> Array.exists (fun arg -> isFreeIn v arg)
       else false

    let betterName (a: string) (b: string) = 
        b.StartsWith("?") && not (a.StartsWith("?"))

    let getEliminationMatrix (solver: Solver) =

        // Find the equations 
        let eqns = solver.Assertions |> Array.filter (fun x -> x.IsEq) |> Array.map (fun x -> x.Args.[0], x.Args.[1])

        // Find the equations defining variables, prefering a nicer name in a = b
        let veqns = 
            eqns 
            |> Array.choose (fun (a,b) -> 
                if a.IsConst && b.IsConst && betterName (a.ToString()) (b.ToString()) then Some(b,a) 
                elif a.IsConst then Some(a,b) 
                elif b.IsConst then Some(b,a) 
                else None)

        // Iteratively find all the equations where the rhs don't use any of the variables, e.g. "x = 1"
        // and normalise the e.h.s. of the other equations with respect to these
        let rec loop (veqns: (Expr * Expr)[]) acc =
            //printfn "veqns = %A"  veqns
            //printfn "acc = %A"  acc
            let relv, others = veqns |> Array.partition (fun (v,rhs) -> not (isFreeIn v rhs))
            //printfn "relv = %A"  relv
            match Array.toList relv with 
            | [] -> Array.ofList (List.rev acc)
            | (relv, rele)::others2 -> 
               //printfn "relv = %A, rele = %A"  relv rele
               let others = Array.append (Array.ofList others2) others
               let others = others |> Array.map (fun (v,b) -> (v.Substitute(relv, rele), b.Substitute(relv, rele)))
               //printfn "others = %A"  others
               loop others ((relv, rele) :: acc)
        loop veqns [ ]


[<RequireQualifiedAccess>]
type Sym(syms: SymContextImpl, z3Expr: Expr) =

    member sym.SymContext = syms

    /// Assert the two symbols to be equal
    member sym1.Solve(sym2: ISym) : bool =
        sym1.SymContext.Assert("eq", [|sym1; sym2|])

    member sym.Z3Expr = z3Expr

    override sym.ToString() = sym.SymContext.Format(sym)

    ///// Check if we have enough information for the two symbols to be equal
    //member sym1.TryKnownToBeEqual(sym2: Sym) : bool voption =
    //    //printfn "------" 
    //    //printfn "TryKnownToBeEqual : %O" sym
    //    if sym1.SymContext.Check("eq", [|sym1; sym2|]) |> snd then
    //        if sym1.SymContext.Check("neq", [|sym1; sym2|]) |> snd then
    //            ValueNone
    //        else
    //            ValueSome true
    //    else
    //        ValueSome false

    interface ISym with 
      member sym.SymContext = (sym:Sym).SymContext :> ISymScope

      member sym.GetVarName() = (sym:Sym).SymContext.GetVarName(sym)

      member sym.TryGetConst() =
          match sym.Z3Expr with 
          | IntNum n -> ValueSome (box n)
          | _ -> ValueNone

[<AutoOpen>]
module SymbolPatterns =
    let (|Sym|) (x: ISym) : Sym = (x :?> Sym)

type SymVar(syms: SymContextImpl, name: string, z3Expr: Expr) =
    member _.SymContext = syms
    member _.Name = name
    member _.Z3Expr = z3Expr
    override _.ToString() = "?" + name

type SymContextImpl() =
    let zctx = new Context()
    let solver = zctx.MkSolver()
    let mutable elimCache = None
    //let zparams = zctx.MkParams()
    //let mutable last = 777000000
    let mapping = ConcurrentDictionary<uint32, string>()
    //let solutions = ConcurrentDictionary<int, ISym>()
    //let constraints = ResizeArray<BoolExpr>()

    member syms.Assert(func: string, args: ISym[]) =
        //printfn "------" 
        let args = args |> Array.map (fun (Sym x) -> x)
        //printfn "constraint: %s(%s)" func (String.concat "," (Array.map string args))
        let expr = syms.Create(func, args)
        let zexpr = expr.Z3Expr :?> BoolExpr
        let res = solver.Check(zexpr)
        match res with
        | Status.UNSATISFIABLE ->
            //printfn "constraint not ok --> %s(%s)" func (String.concat "," (Array.map string args))
            false
        | _ -> 
            //printfn "constraint ok!: %s(%s)" func (String.concat "," (Array.map string args))
            elimCache <- None
            solver.Assert(zexpr)
            true

    member syms.CreateVar (name: string) : Sym =
        let zsym = zctx.MkFreshConst(name, zctx.IntSort)
        //last <- last + 1
        //mapping.[code] <- name
        mapping.[zsym.Id] <- name
        Sym (syms, zsym)

    interface ISymScope with
    
        /// Create a symbol var with the given name and constrain it to be equal to the 
        /// given constant value
        override syms.CreateConst (v: obj) : ISym =
       
           let zsym = 
               match v with 
               | :? int as n -> zctx.MkInt(n) :> Expr
               | :? string as s -> zctx.MkString(s) :> Expr
               | _ -> failwithf "unknown constant %O or type %A" v (v.GetType())
           Sym(syms, zsym) :> ISym

        override syms.CreateApp (f: string, args: ISym[]) : ISym =
            let args = args |> Array.map (fun (Sym x) -> x)
            syms.Create(f, args) :> ISym

        override syms.CreateVar (name: string) : ISym = syms.CreateVar (name) :> ISym
        override syms.Assert(func: string, args: ISym[]) = syms.Assert(func, args)
        override _.Push() = solver.Push()
        override _.Pop() = solver.Pop()

    member syms.Create (f: string, args: Sym[]) : Sym =
        let zsym = 
           match f with 
           | "add" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkAdd(zargs) :> Expr
           | "mul" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkMul(zargs) :> Expr
           | "sub" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkSub(zargs) :> Expr
           | "div" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               //solver.Assert(zctx.MkGt(zargs.[1], zctx.MkInt(0)))
               zctx.MkDiv(zargs.[0], zargs.[1]) :> Expr
           | "mod" -> 
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> IntExpr)
               zctx.MkMod(zargs.[0], zargs.[1]) :> Expr
           //| "max" -> 
           //    let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
           //    zctx.M .MkSub(zargs) :> Expr
           | "leq" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkLe(zargs.[0], zargs.[1]) :> Expr
           | "lt" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkLt(zargs.[0], zargs.[1]) :> Expr
           | "geq" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkGe(zargs.[0], zargs.[1]) :> Expr
           | "gt" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr :?> ArithExpr)
               zctx.MkGt(zargs.[0], zargs.[1]) :> Expr
           | "eq" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr)
               zctx.MkEq(zargs.[0], zargs.[1]) :> Expr
           | "neq" ->
               let zargs = args |> Array.map (fun x -> x.Z3Expr)
               zctx.MkNot (zctx.MkEq(zargs.[0], zargs.[1])) :> Expr
           | s -> 
               printfn "assuming %s is uninterpreted" s
               // TODO: string sorts and others
               let funcDecl = zctx.MkFuncDecl(s,[| for _x in args -> zctx.IntSort :> Sort |], (zctx.IntSort :> Sort))
               let zargs = args |> Array.map (fun x -> x.Z3Expr)
               zctx.MkApp(funcDecl, zargs)
       
        Sym(syms, zsym)

    member syms.Check(func: string, args: Sym[]) : bool =
        
        //printfn "check: %s(%s)" func (String.concat "," (Array.map string args))
        let expr = syms.Create(func, args)
        let zexpr = expr.Z3Expr :?> BoolExpr
        let res = solver.Check(zexpr)
        res <> Status.UNSATISFIABLE

    member _.Solver = solver

    /// Canonicalise an expression w.r.t. the equations in Solver
    member _.EliminateVarEquations (expr: Expr) =
        let elim = 
            match elimCache with 
            | None -> let res = getEliminationMatrix solver in elimCache <- Some res; res
            | Some e -> e
        //printfn "expr = %O, shell = %A"  expr elim
        expr.Substitute(Array.map fst elim, Array.map snd elim)

    member _.GetVarName(sym: Sym) = 
        match mapping.TryGetValue sym.Z3Expr.Id with
        | true, v -> v
        | _ -> "?"

    member _.Format(sym: Sym) = 
        let parenIf c s = if c then "(" + s + ")" else s
        let isNegSummand (s: Expr) =
           match s with 
           | Mul [| IntNum n; _arg|] when n < 0 -> true
           | IntNum n when n < 0 -> true
           | _ -> false
        let rec print prec (zsym: Expr) =
            if zsym.IsAdd then
                // put negative summands at the end
                let args = zsym.Args |> Array.sortBy (fun arg -> if isNegSummand arg then 1 else 0)
                let argsText =
                   args 
                   |> Array.mapi (fun i arg -> 
                       match arg with 
                       | IntNum n when n < 0 -> string n
                       | Mul [| IntNum -1; arg|] -> "-"+print 6 arg
                       | Mul [| IntNum n; arg|] when n < 0 -> "-"+print 6 (zctx.MkMul(zctx.MkInt(-n),(arg :?> ArithExpr)))
                       | _ -> (if i = 0 then "" else "+") + print 6 arg)
                   |> String.concat ""
                parenIf (prec<=3) argsText 
            elif zsym.IsSub then parenIf (prec<=3) (zsym.Args |> Array.map (print 5) |> String.concat "-")
            elif zsym.IsMul then parenIf (prec<=5) (zsym.Args |> Array.map (print 3) |> String.concat "*")
            elif zsym.IsIDiv then parenIf (prec<=5) (zsym.Args |> Array.map (print 3) |> String.concat "/")
            elif zsym.IsRemainder then parenIf (prec<=5) (zsym.Args |> Array.map (print 5) |> String.concat "%")
            elif zsym.IsApp && zsym.Args.Length > 0 then parenIf (prec<=2) (zsym.FuncDecl.Name.ToString() + "(" + (zsym.Args |> Array.map (print 5) |> String.concat ",") + ")")
            elif zsym.IsConst then 
                match mapping.TryGetValue zsym.Id with
                | true, v -> v
                | _ -> zsym.ToString()
            else zsym.ToString()
        let simp = sym.Z3Expr.Simplify()
        let simp2 = simp |> sym.SymContext.EliminateVarEquations
        print 6 simp2

#endif
