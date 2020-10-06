namespace Tests

open System
open DiffSharp
open NUnit.Framework

module Combos =

    // Use these to experiment in your local branch
    //let backends = [ Backend.Reference ]
    //let backends = [ Backend.Torch ]
    //let backends = [ Backend.Reference; Backend.Torch; Backend.Register("TestDuplicate") ]
    //let backends = [ Backend.Reference; Backend.Torch ]
    //let backends = [ Backend.Reference; Backend.Register("TestDuplicate") ]
    //let backends = [ Backend.Register("TestDuplicate") ]
    //let getDevices _ = [ Device.CPU ]
    //let getDevices _ = [ Device.GPU ]
    
    //Use this in committed code
    let backends = [ Backend.Reference; Backend.Torch ]
    let getDevices (deviceType: DeviceType option, backend: Backend option) =
        dsharp.devices(?deviceType=deviceType, ?backend=backend)

    let makeCombos dtypes =
        [ for backend in backends do
            let ds = getDevices (None, Some backend)
            for device in ds do
              for dtype in dtypes do
                yield ComboInfo(defaultBackend=backend, defaultDevice=device, defaultDtype=dtype, defaultFetchDevices=getDevices) ]

    let ShapeChecking =
        [ let devices = [ Device.CPU ]
          for device in devices do
              for dtype in [ Dtype.Float32 ] do
                yield ComboInfo(defaultBackend=Backend.ShapeChecking, defaultDevice=device, defaultDtype=dtype, defaultFetchDevices=(fun _ -> devices)) ]

    /// These runs though all devices, backends and various Dtype
    let Float32 = makeCombos Dtypes.Float32
    let Integral = makeCombos Dtypes.Integral
    let FloatingPoint = makeCombos Dtypes.FloatingPoint
    let UnsignedIntegral = makeCombos Dtypes.UnsignedIntegral
    let SignedIntegral = makeCombos Dtypes.SignedIntegral
    let SignedIntegralAndFloatingPoint = makeCombos Dtypes.SignedIntegralAndFloatingPoint
    let IntegralAndFloatingPoint = makeCombos Dtypes.IntegralAndFloatingPoint
    let Bool = makeCombos Dtypes.Bool
    let IntegralAndBool = makeCombos Dtypes.IntegralAndBool
    let All = makeCombos Dtypes.All

    /// This runs though all devices and backends but leaves the default Dtype
    let AllDevicesAndBackends = 
        [ for backend in backends do
            let ds = getDevices (None, Some backend)
            for device in ds do
              yield ComboInfo(defaultBackend=backend, defaultDevice=device, defaultFetchDevices=getDevices) ]

