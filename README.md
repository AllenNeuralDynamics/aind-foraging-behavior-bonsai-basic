# aind-foraging-behavior-bonsai-basic
Basic analyses of foraging behavior .nwb from Bonsai

See doc [here](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline/tree/main#2-in-code-ocean-this-repo-trigger-computation-co-capsule-foraging_behavior_bonsai_pipeline_trigger-github).


## Steps for updating the analysis
1. Test the code by running within the capsule on an example session
2. Commmit the changes in this capsule
3. (Important!) Hold or terminate the running capsule
4. Do a `reproducible run` in CO
   
   <img width="241" alt="image" src="https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-basic/assets/24734299/ff84641b-e6f7-4428-83b3-a8ad1af2b5b0">

6. (Important!) If the update involves rerunning sessions that previously failed, one should remove the entries in NWB_error.json
7. Rerun the trigger capsule
