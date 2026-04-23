
```
C:.
│   .gitignore
│   new_output.png
│   old_output.png
│   README.md
│   
├───data # data folder
│       poor_battery_images_dataset.zip
│       
└───src
    │   app.py
    │   main.py
    │   
    ├───exps # experimental space: we will write the code to test here
    │       exp.ipynb
    │       temp.ipynb
    │       
    ├───generals # general space: we will place commonly used functions here.
    │       postprocess.py
    │       preprocess.py
    │       utils.py
    │       
    └───gradient_tone_mapping # algorithm space: we will write each algo in a folder like this
            gradient.py
            parameters.py
            tonemap.py
```