namespace rec DiffSharp.Backends.ShapeChecking

#if SYMBOLIC_SHAPES
open System
open System.Reflection
open DiffSharp.ShapeChecking
open Microsoft.Z3

type ShapeCheckingBackendSymbolStatics() =
    inherit BackendSymbolStatics()
    override _.CreateSymContext() = SymContextImpl() :> _

[<AutoOpen>]
module internal Util = 
    let (|HasMethod|_|) (nm: string) (t: Type) =
        match t with 
        | null -> None
        | _ -> 
        match t.GetMethod(nm, BindingFlags.Static ||| BindingFlags.Public ||| BindingFlags.NonPublic) with
        | null -> None
        | m -> Some m

    let rec isFreeIn (v: Expr) (tm: Expr) =
       if v = tm then true
       else if tm.IsApp then tm.Args |> Array.exists (fun arg -> isFreeIn v arg)
       else false

    let betterName (a: string) (b: string) = 
        b.StartsWith("?") && not (a.StartsWith("?"))

    let getGaussianElim (solver: Solver) =

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
        let rec loop (veqns: (Expr * Expr)[])  (vs: Expr[]) acc =
            //printfn "vs = %A, acc = %A"  vs acc
            let relv, irrel = veqns |> Array.partition (fun (_,rhs) -> not (vs |> Array.exists (fun v2 -> isFreeIn v2 rhs)))
            if relv.Length = 0 then acc 
            else 
               let vsnew = Array.map fst relv
               let relv = relv |> Array.map (fun (v,b) -> (v, b.Substitute(Array.map fst acc, Array.map snd acc)))
               loop irrel (Array.except vsnew vs) (Array.append relv acc)
        loop veqns (Array.map fst veqns) [| |]

    /// Canonicalise an expression w.r.t. the equations in Solver
    ///
    /// TODO: cache the 'getGaussianElim' though it needs to be updated after each new equation
    let canonicalize (solver: Solver) (expr: Expr) =
        let shell = getGaussianElim solver
        expr.Substitute(Array.map fst shell, Array.map snd shell)

[<RequireQualifiedAccess>]
type Sym =
    | Const of syms: SymContextImpl * value: obj * z3Expr: Expr // e.g. a constant integer, Dtype, Device
    | Var of var: SymVar
    | App of syms: SymContextImpl * func: string * args: Sym[]  * z3Expr: Expr

    member sym.SymContext =
        match sym with
        | Const (syms, _, _) -> syms
        | Var v -> v.SymContext
        | App(syms, _, _, _) -> syms

    /// Constrain the two symbols to be equal
    member sym1.Solve(sym2: ISym) : bool =
        sym1.SymContext.Constrain("eq", [|sym1; sym2|])

    member sym.Z3Expr =
        match sym with 
        | Const (_, _, a) -> a
        | Var v -> v.Z3Expr
        | App (_, _, _, a) -> a

    override sym.ToString() = 
        let parenIf c s = if c then "(" + s + ")" else s
        let rec print prec (s: Expr) =
            if s.IsAdd then parenIf (prec<=3) (s.Args |> Array.map (print 5) |> String.concat "+")
            elif s.IsMul then parenIf (prec<=5) (s.Args |> Array.map (print 3) |> String.concat "*")
            elif s.IsIDiv then parenIf (prec<=5) (s.Args |> Array.map (print 3) |> String.concat "/")
            elif s.IsSub then parenIf (prec<=5) (s.Args |> Array.map (print 5) |> String.concat "-")
            elif s.IsRemainder then parenIf (prec<=5) (s.Args |> Array.map (print 5) |> String.concat "%")
            elif s.IsApp && s.Args.Length > 0 then parenIf (prec<=2) (s.FuncDecl.Name.ToString() + "(" + (s.Args |> Array.map (print 5) |> String.concat ",") + ")")
            else s.ToString()
        let simp = sym.Z3Expr.Simplify() |> canonicalize sym.SymContext.Solver
        print 6 simp
        //match sym with
        //| Const (_, v, _) -> v.ToString()
        //| Var v -> v.Name
        //| App(_, f, args, _) -> f + "(" + String.concat "," (Array.map string args) + ")"

    /// Check if we have enough information for the two symbols to be equal
    member sym1.TryKnownToBeEqual(sym2: Sym) : bool voption =
        //printfn "------" 
        //printfn "TryKnownToBeEqual : %O" sym
        if sym1.SymContext.CheckIfSatisfiable("eq", [|sym1; sym2|]) |> snd then
            if sym1.SymContext.CheckIfSatisfiable("neq", [|sym1; sym2|]) |> snd then
                ValueNone
            else
                ValueSome true
        else
            ValueSome false

    interface ISym with 
      member sym.SymContext = (sym:Sym).SymContext :> SymContext

      member sym1.TryKnownToBeEqual(sym2: ISym) = sym1.TryKnownToBeEqual(sym2 :?> Sym)

      member sym.GetVarName() =
        match sym with
        | Var v -> v.Name
        | _ -> printfn "not a variable"; "?"

      member sym.TryEvaluate() = ValueNone

[<AutoOpen>]
module SymbolPatterns =
    let (|Sym|) (x: ISym) : Sym = (x :?> Sym)

type SymVar(syms: SymContextImpl, name: string, z3Expr: Expr) =
    member _.SymContext = syms
    member _.Name = name
    member _.Z3Expr = z3Expr
    override _.ToString() = "?" + name

type SymContextImpl() =
    inherit SymContext()
    
    let zctx = new Context()
    let solver = zctx.MkSolver()
    //let zparams = zctx.MkParams()
    //let mutable last = 777000000
    //let mapping = ConcurrentDictionary<int, string>()
    //let solutions = ConcurrentDictionary<int, ISym>()
    //let constraints = ResizeArray<BoolExpr>()

    /// Create a symbol var with the given name and constrain it to be equal to the 
    /// given constant value
    override syms.CreateConst (v: obj) : ISym =
       
       let zsym = 
           match v with 
           | :? int as n -> zctx.MkInt(n) :> Expr
           | :? string as s -> zctx.MkString(s) :> Expr
           | _ -> failwithf "unknown constant %O or type %A" v (v.GetType())
       Sym.Const(syms, v, zsym) :> ISym

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
       
        Sym.App(syms, f, args, zsym)

    override syms.CreateApp (f: string, args: ISym[]) : ISym =
        let args = args |> Array.map (fun (Sym x) -> x)
        syms.Create(f, args) :> ISym

    override syms.CreateVar (name: string) : ISym =
        let zsym = zctx.MkConst(name, zctx.IntSort)
        let sym = SymVar(syms, name, zsym)
        //last <- last + 1
        //mapping.[code] <- name
        Sym.Var sym :> ISym

    /// Injects a symbol into a .NET type via a call to .Symbolic on type or partner module.
    override syms.CreateInjected<'T> (name: string) : 'T =
        let sym = syms.CreateVar (name)
        let t = typeof<'T>

        match typeof<'T> with
        | HasMethod "Symbolic" m ->
            m.Invoke(null, [| box sym |]) |> unbox
        | _ -> 
        match t.Assembly.GetType(t.FullName + "Module") with
        | HasMethod "Symbolic" m ->
             m.Invoke(null, [| box sym |]) |> unbox
        | _ ->
            failwithf "no static 'ShapeChecking' method found in type '%s' or a partner module of the same name" t.FullName

    override ctxt.Constrain(func: string, args: ISym[]) =
        //printfn "------" 
        let args = args |> Array.map (fun (Sym x) -> x)
        //printfn "constraint: %s(%s)" func (String.concat "," (Array.map string args))
        let zexpr, res = ctxt.CheckIfSatisfiable(func, args)
        if res then 
            //printfn "constraint ok!: %s(%s)" func (String.concat "," (Array.map string args))
            solver.Assert(zexpr)
        //else
            //printfn "constraint not ok --> %s(%s)" func (String.concat "," (Array.map string args))
        //constraints.Add(zexpr)
        res

    member syms.CheckIfSatisfiable(func: string, args: Sym[]) : BoolExpr * bool =
        
        //printfn "check: %s(%s)" func (String.concat "," (Array.map string args))
        let expr = syms.Create(func, args)
        let zexpr = expr.Z3Expr :?> BoolExpr
        solver.Push()
        solver.Assert(zexpr)
        let res = solver.Check()
        solver.Pop()
        match res with
        | Status.UNSATISFIABLE -> 
            //printfn "  --> not ok"
            zexpr, false
        | Status.SATISFIABLE -> 
            //printfn "  --> ok"
            zexpr, true
        | _ -> 
            //printfn "  --> unknown"
            zexpr, true

    member _.Solver = solver
#endif
