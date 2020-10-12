namespace DiffSharp.ShapeChecking

open DiffSharp

type SourceLocation = 
   { File: string
     StartLine: int 
     StartColumn: int 
     EndLine: int 
     EndColumn: int }
   override loc.ToString() = sprintf "%s: (%d,%d)-(%d,%d)" loc.File loc.StartLine loc.StartColumn loc.EndLine loc.EndColumn

type Diagnostic =
   { Severity: int
     Number: int
     Message: string
     LocationStack: SourceLocation[] }
   member x.Location = Array.last x.LocationStack

[<AutoOpen>]
module ShapeCheckingAutoOpens =
    type ISymScope with 
        member syms.CreateFreshIntVar(name:string, ?location:SourceLocation) =
            Int.FromSymbol (syms.CreateFreshVar(name, location=Option.toObj (Option.map box location)))

        member syms.CreateIntVar(name:string, ?location:SourceLocation) =
            Int.FromSymbol (syms.CreateVar(name, location=Option.toObj (Option.map box location)))

        /// Create an inferred symbol 
        member syms.Infer = syms.CreateFreshIntVar("?")

    /// Create a symbol in the global symbol context of the given name
    let (?) (syms: ISymScope) (name: string) : Int = syms.CreateIntVar(name)

    // Shen using shape checking the syntax 128I is hijacked
    module NumericLiteralI = 
        let FromZero () : Int = Int 0
        let FromOne () : Int = Int 1
        let FromInt32 (value:int32): Int = Int value
        

