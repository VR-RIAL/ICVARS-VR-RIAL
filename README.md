# ICVARS-VR-RIAL
Virtual Reality for Rule-Infused Architectural Layouts

Usage:
1. Access ```Mistral AI``` through an API for ```input_parsing.py``` and ```output_parsing.py```

2. Run ```python3 input_parsing.py```. It will format the textual input into a JSON format.

3. Run ```python3 main.py Inputs/{directory_number}``` to generate the 2D layout. Once completed, you will be asked if you want to visualise it in the 3D space.

4. Run ```python3 output_parsing.py``` to generate a json containing all extractable information from the generated image.

5. Run ```python3 final_accuracy_checker.py``` to compare input requirements to output image to determine if the pipeline has missed anything.
