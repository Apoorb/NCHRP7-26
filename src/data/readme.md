## List of Files

1. output_pre_breakdown_flow_speed_updated.py: 
   Copy of Azy's script for generating csv files for pre-breakdown and uncongested volumes. 
   Some of the things in this script are hard coded; for instance the FFS fixed parameter 
   is read from NCHRP07-26_Site_summary_shared.xlsx.
   
    **Check with Azy (if possible) before using this script!**

    He might have made some changes to his version of this script that would give different
    output.
    
    **If possible let Azy generate the ouput of this script.**

    The outputs of this script includes the following two files with following columns:
        1_Simple Merge_1_pre_brkdn.csv: Time, MainlineSpeed, MainlineVol
            This file has the pre-breakdown data.
        1_Simple Merge_1_uncongested.csv: Time, MainlineSpeed, MainlineVol
            This file has the uncongested data (which includes the pre-breakdown data).
   
2. clean_prebreakdown_data.py: Script to combine all the uncongested and pre-breakdown 
   data by merge, diverge, and weave sites. Also add FFS from Azy's Data_output-Final.csv 
   file and add metadata from the NCHRP 07-26_Master_Database_shared.xlsx file.
