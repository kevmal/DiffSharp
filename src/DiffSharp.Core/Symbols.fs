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
    abstract CreateSymContext: unit -> ISymScope

    /// Get the implementation for the given backend.
    static member Get() =
        hook.Get(Backend.ShapeChecking)

type NoBackendSymbolStatics() = 
    inherit BackendSymbolStatics()
    override _.CreateSymContext() = failwith "no symbol scope"

type ISym =
    abstract SymContext : ISymScope

    /// Get the name of a symbol that is a variable
    abstract GetVarName : unit -> string

    /// Try to get the symbol to a constant
    abstract TryGetConst : unit -> obj voption

[<AutoOpen>]
module SymbolExtensions =

    type ISym with   

        static member unop nm (arg: ISym) : ISym =
            arg.SymContext.CreateApp(nm, [|arg|])

        static member binop nm (arg1: ISym) (arg2: ISym) : ISym =
            arg1.SymContext.CreateApp(nm, [|arg1; arg2|])

        static member app nm (args: ISym []) : ISym =
            args.[0].SymContext.CreateApp(nm, args)

        /// Assert the two symbols to be equal
        member sym1.Solve(sym2) =
            sym1.SymContext.Assert("eq", [|sym1; sym2|])

/// Represents an accumulating collection of related symbols and constraints
type ISymScope =

    /// Create a symbol var with the given name and constrain it to be equal to the 
    /// given constant value
    abstract CreateConst: v: obj -> ISym 

    /// Create an application symbol
    abstract CreateApp: func: string * args: ISym[] -> ISym 

    /// Create a variable symbol
    abstract CreateVar: name: string -> ISym

    /// Asserts a constraint in the solver state
    abstract Assert: func: string * args: ISym[]  -> bool

    /// Checkpoint the solver state
    abstract Push: unit -> unit

    /// Revert the solver state
    abstract Pop: unit -> unit

//[<AutoOpen>]
//module SymbolsAutoOpens =
    
//    // An active pattern to check for a solution to a symbol
//    let (|Solved|_|) (sym: ISym) : ISym option = 
//        match sym.TryGetSolution() with
//        | ValueNone -> None
//        | ValueSome sym -> Some sym
//let v : int = (sym?A) 

#endif
