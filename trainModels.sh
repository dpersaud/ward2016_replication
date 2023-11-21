#!/bin/bash
# Run this file using:
#       ./trainModels.sh
# Ensure version 1.8 of java is running
# Check version using:
#       java -version
# If you have version 1.8 installed, run the following on mac or linux to activate version 1.8:
#       export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)

# Define a list of random seeds
lstRandomSeed=(0 1 15 42 58 66 73 152 982 8653)

# Define the template script with a placeholder
template_script="template-train-model.in"
cp train-model.in "$template_script"

# Loop through the list
for intRandomSeed_temp in "${lstRandomSeed[@]}"; do
  # Replace the placeholder with the current random seed
  awk -v seed="$intRandomSeed_temp" '{gsub("RANDOMSEED_placeholder", seed)}1' "$template_script" > "train-model-$intRandomSeed_temp.in"

  # Run the Java command with the modified script
  java -jar magpie/dist/Magpie.jar "train-model-$intRandomSeed_temp.in" > "models/trainedModels/logs/train-model-seed$intRandomSeed_temp.out"

  # Remove the temporary input script
  rm "train-model-$intRandomSeed_temp.in"

  # Print a message to the screen
  echo "Finished training model with random seed $intRandomSeed_temp"
done

# Remove the temporary template file
rm "$template_script"

