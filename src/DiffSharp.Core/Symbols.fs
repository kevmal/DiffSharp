namespace rec DiffSharp.ShapeChecking

#if SYMBOLIC_SHAPES
open DiffSharp

/// <summary>
///   Represents the static functionality for symbolic shape checking implemented by a DiffSharp backend.
/// </summary>
[<AbstractClass>]
type BackendSymbolStatics() = 
    static let hook = BackendFunctionality<BackendSymbolStatics>()

    /// Sets the seed for the default random number generator of the backend
    abstract GetSymbolScope: unit -> SymbolScope

    /// Get the implementation for the given backend.
    static member Get(?backend: Backend) =
        let backend = defaultArg backend Backend.Default
        hook.Get(backend)

type NoBackendSymbolStatics() = 
    inherit BackendSymbolStatics()
    override _.GetSymbolScope() = failwith "no symbol scope"

type Symbol =
    abstract SymbolScope : SymbolScope

    /// Check if we have enough information for the two symbols to be equal
    abstract TryKnownToBeEqual: sym2: Symbol -> bool voption

    /// Constrain the two symbols to be equal
    abstract Solve: sym2: Symbol -> bool 

    /// Get the name of a symbol that is a variable
    abstract GetVarName : unit -> string

    /// Get the unique id of a symbol that is a variable
    abstract GetVarId : unit -> int

    /// Determine if the symbol is a variable with a known equational solution
    abstract TryGetSolution : unit -> Symbol voption

    /// Try to evaluate the symbol to a constant
    abstract TryEvaluate : unit -> obj voption

[<AutoOpen>]
module SymbolExtensions =

    type Symbol with   
        member sym1.KnownToBeEqual(sym2: Symbol) : bool =
            match sym1.TryKnownToBeEqual(sym2) with 
            | ValueNone -> false
            | ValueSome v -> v

        static member unop nm (arg: Symbol) : Symbol =
            arg.SymbolScope.CreateApp(nm, [arg])

        static member binop nm (arg1: Symbol) (arg2: Symbol) : Symbol =
            arg1.SymbolScope.CreateApp(nm, [arg1; arg2])

        static member app nm (args: Symbol list) : Symbol =
            args.[0].SymbolScope.CreateApp(nm, args)

[<AbstractClass>]
/// Represents an accumulating collection of related symbols and constraints
type SymbolScope() =

    /// Create a symbol var with the given name and constrain it to be equal to the 
    /// given constant value
    abstract CreateConst: v: obj -> Symbol 

    /// Create an application symbol
    abstract CreateApp: func: string * args: Symbol list -> Symbol 

    /// Create a variable symbol
    abstract CreateVar: name: string -> Symbol

    /// Injects a symbol into a .NET type via a call to .Symbolic on type or partner module.
    abstract CreateInjected<'T> : name: string -> 'T 

    abstract Constrain: func: string * args: Symbol list -> bool

    /// <summary>Allows <c>sym?Name</c> notation for symbols in test code, avoiding lots of pesky strings</summary>
    static member op_Dynamic (x: SymbolScope, name: string) : 'T = x.CreateInjected<'T>(name)

[<AutoOpen>]
module SymbolsAutoOpens =
    
    // An active pattern to check for a solution to a symbol
    let (|Solved|_|) (sym: Symbol) : Symbol option = 
        match sym.TryGetSolution() with
        | ValueNone -> None
        | ValueSome sym -> Some sym
//let v : int = (sym?A) 

#endif
