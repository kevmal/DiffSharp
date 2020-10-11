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
     Location: SourceLocation }

[<AutoOpen>]
module ShapeCheckingAutoOpens =
    type ISymScope with 
        member syms.CreateFreshIntVar(name:string, ?location:SourceLocation) =
            Int.FromSymbol (syms.CreateFreshVar(name, location=Option.toObj (Option.map box location)))

        //member syms.DeviceVar(name:string) =
        //    let dt = LanguagePrimitives.EnumOfValue (hash (s.GetVarName()))
        //    DeviceType.Symbolic(sym.SymScope.CreateVar(sym.GetVarName()))
        //    let device = Device(dt, 0)
        //    device

        member syms.CreateIntVar(name:string, ?location:SourceLocation) =
            Int.FromSymbol (syms.CreateVar(name, location=Option.toObj (Option.map box location)))

    /// Create a symbol in the global symbol context of the given name
    let (?) (syms: ISymScope) (name: string) : Int = syms.CreateIntVar(name)
