import text_analysis_v8
import tfidf_annif
import combine_sheets

check_raw = input('Please enter if you are using raw metadata export(s) directly downloaded from R3 (yes/ no)\n')

if check_raw == 'yes' or check_raw == 'Yes' or check_raw == 'y':
    R3_raw = input('Please enter your metadata export(s) file location. Such as: C:/Users/Documents/Project\n' )
    R3_file = combine_sheets.combine_R3(R3_raw)
    exit()
else:
    R3_file = input('Please enter your R3 metadata file name, including the .csv\n')

    

basic_class = input('Please enter if you would like to start with the main classifier or skip to using Annif on its own (yes/ no)\n')
if basic_class == 'yes' or basic_class == 'Yes' or basic_class == 'y':
    external_file = input('Please enter your external training set file name, including the .csv\n')
    
    output_self = input('Please enter how would you like to name the result trained on R3 data alone, including the .csv\n')
    
    text_analysis_v8.R3_classify(R3_file,output_self)
    cleaned_result = text_analysis_v8.External_classify(external_file,output_self)
    
    use_annif = input('Please enter if you would like to engage Annif to provide more suggestion on potentially relavent subjects, note that this module can take a while to run (yes/no)\n')
    if use_annif == 'yes' or use_annif == 'Yes' or use_annif == 'y':
        tfidf_annif.Annif_classify(cleaned_result, external_file)
        
    else:
        print('You selected not to engage Annif. If you want to engage Annif yourself, call tfidf_annif.Annif_classify(*your_metadata.csv*) in the command line, change the *your_metadata.csv* to whatever metadata you wish to run Annif on.\n')

else:
    print('You indicated that you wish to skip the main classifier, so only the Annif classifier would be called. If you mistyped, rerun the script.\n')
    use_class = input('Please enter if you would like to use the main classifier to assign subclass to entries based on Annif labels (yes/no)\n')
    if use_class == 'yes' or use_class == 'Yes' or use_class == 'y':
        external_file = input('Please enter your external training set file name, including the .csv\n')
        text_analysis_v8.External_only(external_file)
        tfidf_annif.Annif_classify(R3_file, external_file)
    else:
        tfidf_annif.Annif_classify(R3_file)