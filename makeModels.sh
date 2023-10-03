#!/bin/bash

# Define a list of random seeds
lstRandomSeed=(0 1 15 42 58 66 73 152 982 8653)

# Define the template script with a placeholder
template_script="template-hierarchical-bandgap-model.in"
cp hierarchical-bandgap-model.in "$template_script"

# Loop through the list
for intRandomSeed_temp in "${lstRandomSeed[@]}"; do
  # Replace the placeholder with the current random seed
  awk -v seed="$intRandomSeed_temp" '{gsub("RANDOMSEED_placeholder", seed)}1' "$template_script" > "hierarchical-bandgap-model-$intRandomSeed_temp.in"

  # Run the Java command with the modified script
  java -jar magpie/dist/Magpie.jar "hierarchical-bandgap-model-$intRandomSeed_temp.in" > "models/baseModels/logs/hierarchical-bandgap-model-seed$intRandomSeed_temp.out"

  # Remove the temporary input script
  rm "hierarchical-bandgap-model-$intRandomSeed_temp.in"

  # Print a message to the screen
  echo "Finished creating model with random seed $intRandomSeed_temp"
done

# Remove the temporary template file
rm "$template_script"

