# ECD323-Technical-Manual
ECD323 Visual Layer Attacks Detection Documention:
- PCB Design
- 3D-Printed Housing
- Functional/Demonstration Programs

3D-Printed Housing:

First, decide which prototype you will be printing. To print the housing, download the STL files for the lid, "NLid.stl" for prototype 1 and "NLid2.stl" for prototype 2, and base, "NBase.stl" for prototype 1 and "NBase2.stl" for prototype 2. Import them into a 3D printing slicer, such as the Prusa Slicer (https://www.prusa3d.com/page/prusaslicer_424/). Once imported, selected the type of filament, which is Generic PLA, and the type of printer being used. For this project, we used a Prusa MK3S+ 3D printer. Select the option to add a brim. The rest of the settings in the slicer should remain the same. Slice the file, generate the gcode for the file, and then export the gcode from the slicer to the SD card of the 3D printer. The files should now be ready for printing. 

PCD Design Reproduction:
1. Order the PCBs through a vendor using the gerber files. To access the gerber files of the desired design, unzip the appropriate folder and navigating to the gerber files which lie within another zipped folder
2. Order the circuit components from the bill of materials off Digikey
3. Solder components to the PCB
4. Integrate the PCB into the design by connecting the power cord and audio jack
