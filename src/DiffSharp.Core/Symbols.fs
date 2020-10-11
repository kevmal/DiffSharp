namespace rec DiffSharp.ShapeChecking

#if SYMBOLIC_SHAPES
open DiffSharp

type ISym =
    abstract SymScope : ISymScope

    /// Get the name of a symbol that is a variable
    abstract GetVarName : unit -> string

    /// Try to get the symbol to a constant
    abstract TryGetConst : unit -> obj voption

[<AutoOpen>]
module SymbolExtensions =

    type ISym with   

        static member unop nm (arg: ISym) : ISym =
            arg.SymScope.CreateApp(nm, [|arg|])

        static member binop nm (arg1: ISym) (arg2: ISym) : ISym =
            arg1.SymScope.CreateApp(nm, [|arg1; arg2|])

        static member app nm (args: ISym []) : ISym =
            args.[0].SymScope.CreateApp(nm, args)

        /// Assert the two symbols to be equal
        member sym1.AssertEqualityConstraint(sym2) =
            sym1.SymScope.AssertConstraint("eq", [|sym1; sym2|])

/// Represents an accumulating collection of related symbols and constraints
type ISymScope =

    /// Create a symbol var with the given name and constrain it to be equal to the 
    /// given constant value
    abstract CreateConst: v: obj -> ISym 

    /// Create an application symbol
    abstract CreateApp: func: string * args: ISym[] -> ISym 

    /// Create a variable symbol, identical to any other symbol of the same type in this scope,
    /// attaching the given additional information to the variable, e.g. a location
    abstract CreateVar: name: string * location: obj -> ISym

    /// Create a variable symbol, distinct from any other symbol of the same type in this scope,
    /// attaching the given additional information to the variable, e.g. a location
    abstract CreateFreshVar: name: string * location: obj -> ISym

    /// Asserts a constraint in the solver state, returning true if the constraint is consistent
    /// with the solver state, and false if it is inconsistent.
    abstract AssertConstraint: func: string * args: ISym[]  -> bool

    /// Check the satisfiability of a constraint in the solver state without changing solver state.
    /// Return true if the constraint is consistent with the solver state, and false if it is inconsistent.
    abstract CheckConstraint: func: string * args: ISym[]  -> bool

    /// Checkpoint the solver state
    abstract Push: unit -> unit

    /// Revert the solver state
    abstract Pop: unit -> unit

    /// Clear the solver state
    abstract Clear: unit -> unit

    /// Report a diagnostic related to this set of symbols and their constraints.
    /// Severity is 0=Informational, 1=Warning, 2=Error.
    abstract ReportDiagnostic: severity: int * message: string -> unit

#endif
