def my_cad_function(args):
    import cadquery as cq
    
    # Load the original part
    shape = cq.importers.importStep(args["input_file"])
    
    # Create a 1.5mm thick rib for structural support
    # The rib will run along the center of the part in the X direction
    # and extend upward from the base (y=-0.01) to provide vertical support
    
    # Rib dimensions based on measured geometry:
    # - Thickness: 1.5 mm (as requested)
    # - Length: approximately 8 mm (central portion of part)
    # - Height: approximately 3 mm (extending from base upward)
    
    # Position the rib at the center of the part
    rib_center_x = 6.15  # near part center
    rib_center_z = 0.2   # near part center in Z
    
    # Create the rib as a thin vertical plate
    rib = (cq.Workplane("XY")
           .workplane(offset=-0.01)  # Start at base plane y=-0.01
           .center(rib_center_x, rib_center_z)
           .rect(8, 1.5)  # 8mm long, 1.5mm thick
           .extrude(3)    # 3mm tall
          )
    
    # Union the rib with the original part
    result = shape.union(rib)
    
    print(f"Added 1.5mm thick rib for structural support")
    print(f"Rib position: centered at x={rib_center_x}, z={rib_center_z}")
    print(f"Rib dimensions: 8mm x 1.5mm x 3mm (L x thickness x H)")
    
    return result
