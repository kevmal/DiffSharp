namespace rec DiffSharp.Backends.ShapeChecking

#if SYMBOLIC_SHAPES
open System
open System.Reflection
open System.Collections.Concurrent
open DiffSharp
open DiffSharp.ShapeChecking

type ShapeCheckingBackendSymbolStatics() =
    inherit BackendSymbolStatics()
    override _.GetSymbolScope() = SymbolScopeImpl() :> _

[<AutoOpen>]
module internal Util = 
    let (|HasMethod|_|) (nm: string) (t: Type) =
        match t with 
        | null -> None
        | _ -> 
        match t.GetMethod(nm, BindingFlags.Static ||| BindingFlags.Public ||| BindingFlags.NonPublic) with
        | null -> None
        | m -> Some m

[<RequireQualifiedAccess>]
type SymbolImpl =
    | Const of syms: SymbolScope * obj // e.g. a constant integer, Dtype, Device
    | Var of var: SymbolVar
    | App of syms: SymbolScope * func: string * args: Symbol list

    interface Symbol with 
      member sym.SymbolScope =
        match sym with
        | Const (syms, _) -> syms
        | Var v -> v.SymbolScope
        | App(syms, _, _) -> syms

    /// Check if we have enough information for the two symbols to be equal
      member sym1.TryKnownToBeEqual(sym2: Symbol) : bool voption =
        match sym1, (sym2 :?> SymbolImpl) with
        | Const (_, obj1), Const (_, obj2) -> ValueSome (obj1.Equals(obj2))
        | Solved sym1, _ -> sym1.TryKnownToBeEqual(sym2)
        | _, Solved sym2 -> (sym1 :> Symbol).TryKnownToBeEqual(sym2)
        | Var v1, Var v2 -> if (v1 = v2) then ValueSome true else ValueNone
        | App(_, f1, args1), App(_, f2, args2) when f1 = f2 ->
            (ValueSome true, args1, args2) |||> List.fold2 (fun v arg1 arg2 -> if v = ValueSome true then arg1.TryKnownToBeEqual(arg2) else v)
        | a, b ->
            printfn "unsure of %A = %A" a b
            ValueNone

      /// Constrain the two symbols to be equal
      member sym1.Solve(sym2: Symbol) : bool =
        match sym1, (sym2 :?> SymbolImpl) with
        | Const (_, obj1), Const (_, obj2) -> obj1.Equals(obj2)
        | Var v1, _ -> v1.Solve(sym2)
        | _, Var v2 -> v2.Solve(sym1)
        // This only applis if 'f1' is equation-free
        //| App(_, f1, args1), App(_, f2, args2) when f1 = f2 && args1.Length = args2.Length ->
        //    (args1, args2) ||> List.forall2 (fun arg1 arg2 -> arg1.Solve(arg2))
        | a, b ->
            // TODO the real work
            printfn "equation: %O = %O" a b
            true

      member sym.GetVarName() =
        match sym with
        | Var v -> v.Name
        | _ -> failwith "not a variable"

      member sym.GetVarId() =
        match sym with
        | Var v -> v.Id
        | _ -> failwith "not a variable"

      member sym.TryGetSolution() : Symbol voption =
        match sym with
        | Const (_, _) -> ValueNone
        | Var v -> v.TryGetSolution()
        | App(_, _, _) -> ValueNone

      member sym.TryEvaluate() =
        match sym with 
        | SymbolImpl.Const (_, (:? int as n)) -> ValueSome (box n)
        | Solved sym -> sym.TryEvaluate()
        | SymbolImpl.App(_, "add", [a;b]) ->  
            match a.TryEvaluate(), b.TryEvaluate() with
            | ValueSome (:? int as av), ValueSome (:? int as bv) -> ValueSome (box (av + bv))
            | _ -> ValueNone
        | SymbolImpl.App(_, "sub", [a;b]) ->  
            match a.TryEvaluate(), b.TryEvaluate() with
            | ValueSome (:? int as av), ValueSome (:? int as bv) -> ValueSome (box (av - bv))
            | _ -> ValueNone
        | SymbolImpl.App(_, "mul", [a;b]) ->  
            match a.TryEvaluate(), b.TryEvaluate() with
            | ValueSome (:? int as av), ValueSome (:? int as bv) -> ValueSome (box (av * bv))
            | _ -> ValueNone
        | SymbolImpl.App(_, "div", [a;b]) ->  
            match a.TryEvaluate(), b.TryEvaluate() with
            | ValueSome (:? int as av), ValueSome (:? int as bv) -> ValueSome (box (av / bv))
            | _ -> ValueNone
        | SymbolImpl.App(_, "mod", [a;b]) ->  
            match a.TryEvaluate(), b.TryEvaluate() with
            | ValueSome (:? int as av), ValueSome (:? int as bv) -> ValueSome (box (av % bv))
            | _ -> ValueNone
        | SymbolImpl.App(_, "neg", [a]) ->  
            match a.TryEvaluate() with
            | ValueSome (:? int as av) -> ValueSome (box (-av))
            | _ -> ValueNone
        | _ -> ValueNone

    override sym.ToString() =
        match sym with
        | Const (_, v) -> v.ToString()
        | Var v -> v.Name
        | App(_, f, args) -> f + "(" + String.concat "," (List.map string args) + ")"


type SymbolVar(syms: SymbolScope, code: int, name: string) =
    member _.SymbolScope = syms
    member _.Id = code   
    member _.Name = name
    override _.ToString() = "?" + name
    member v.Solve(sym2:Symbol) = (syms :?> SymbolScopeImpl).Solve(v, sym2)
    member v.TryGetSolution() : Symbol voption = (syms :?> SymbolScopeImpl).TryGetSolution(v)

type SymbolScopeImpl() =
    inherit SymbolScope()
    
    let mutable last = 777000000
    let mapping = ConcurrentDictionary<int, string>()
    let solutions = ConcurrentDictionary<int, Symbol>()
    let constraints = ConcurrentQueue<(string * Symbol list)>()

    /// Create a symbol var with the given name and constrain it to be equal to the 
    /// given constant value
    override syms.CreateConst (v: obj) : Symbol =
       SymbolImpl.Const(syms, v) :> Symbol

    override syms.CreateApp (f: string, args: Symbol list) : Symbol =
       SymbolImpl.App(syms, f, args) :> Symbol

    override syms.CreateVar (name: string) : Symbol =
        let code = last
        let sym = SymbolVar(syms, code, name)
        last <- last + 1
        mapping.[code] <- name
        SymbolImpl.Var sym :> Symbol

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

    override _.Constrain(func: string, args: Symbol list) =
        printfn "constraint: %s(%s)" func (String.concat "," (List.map string args))
        constraints.Enqueue((func, args))
        true

    member _.Solve(v1: SymbolVar, sym2: Symbol) =
        match v1.TryGetSolution() with 
        | ValueSome sym1 -> sym1.Solve(sym2)
        | ValueNone _ -> 
            solutions.[v1.Id] <- sym2
            true

    member _.TryGetSolution(v: SymbolVar) =
        match solutions.TryGetValue(v.Id) with 
        | true, res -> ValueSome res
        | _ -> ValueNone
#endif
