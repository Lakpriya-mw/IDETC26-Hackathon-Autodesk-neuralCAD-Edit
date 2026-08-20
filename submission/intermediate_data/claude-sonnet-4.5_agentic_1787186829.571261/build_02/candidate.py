def my_cad_function(args):
    import cadquery as cq
    
    # Load the original part
    shape = cq.importers.importStep(args["input_file"])
    
    # Create a 1.5mm thick supporting rib
    # Build it as a simple vertical wall from the base plane
    try:
        rib = (cq.Workplane("XZ")
               .workplane(offset=-0.01)
               .center(6.0, 0.0)
               .rect(6.0, 4.0)
               .extrude(1.5)
              )
        
        result = shape.union(rib)
        print(f"Added 1.5mm structural rib: 6mm x 4mm x 1.5mm")
        
    except Exception as e:
        print(f"Union failed: {e}")
        rib = (cq.Workplane("XZ")
               .workplane(offset=0.5)
               .center(6.0, 0.5)
               .rect(4.0, 2.0)
               .extrude(1.5)
              )
        result = shape.union(rib)
        print(f"Added smaller 1.5mm rib")
    
    return result