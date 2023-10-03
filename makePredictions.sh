#!/bin/bash

# Define a list of random seeds
lstRandomSeed=(0 1 15 42 58 66 73 152 982 8653)

# Define the template script with a placeholder
template_script="template-make-predictions.in"
cp make-predictions.in "$template_script"

# Loop through the list
for intRandomSeed_temp in "${lstRandomSeed[@]}"; do
  # Replace the placeholder with the current random seed
  awk -v seed="$intRandomSeed_temp" '{gsub("RANDOMSEED_placeholder", seed)}1' "$template_script" > "make-predictions-$intRandomSeed_temp.in"

  # Run the Java command with the modified script
  java -jar magpie/dist/Magpie.jar "make-predictions-$intRandomSeed_temp.in" > "predictions/logs/make-predictions-seed$intRandomSeed_temp.out"

  # Remove the temporary input script
  rm "make-predictions-$intRandomSeed_temp.in"

  # Print a message to the screen
  echo "Finished predictions using model with random seed $intRandomSeed_temp"
done

# Remove the temporary template file
rm "$template_script"
