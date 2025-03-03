#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/calculates_results_stats.py
#                                                                             
# PROGRAMMER:
# DATE CREATED:                                  
# REVISED DATE: 
# PURPOSE: Create a function calculates_results_stats that calculates the 
#          statistics of the results of the programrun using the classifier's model 
#          architecture to classify the images. This function will use the 
#          results in the results dictionary to calculate these statistics. 
#          This function will then put the results statistics in a dictionary
#          (results_stats_dic) that's created and returned by this function.
#          This will allow the user of the program to determine the 'best' 
#          model for classifying the images. The statistics that are calculated
#          will be counts and percentages. Please see "Intro to Python - Project
#          classifying Images - xx Calculating Results" for details on the 
#          how to calculate the counts and percentages for this function.    
#         This function inputs:
#            -The results dictionary as results_dic within calculates_results_stats 
#             function and results for the function call within main.
#         This function creates and returns the Results Statistics Dictionary -
#          results_stats_dic. This dictionary contains the results statistics 
#          (either a percentage or a count) where the key is the statistic's 
#           name (starting with 'pct' for percentage or 'n' for count) and value 
#          is the statistic's value.  This dictionary should contain the 
#          following keys:
#            n_images - number of images
#            n_dogs_img - number of dog images
#            n_notdogs_img - number of NON-dog images
#            n_match - number of matches between pet & classifier labels
#            n_correct_dogs - number of correctly classified dog images
#            n_correct_notdogs - number of correctly classified NON-dog images
#            n_correct_breed - number of correctly classified dog breeds
#            pct_match - percentage of correct matches
#            pct_correct_dogs - percentage of correctly classified dogs
#            pct_correct_breed - percentage of correctly classified dog breeds
#            pct_correct_notdogs - percentage of correctly classified NON-dogs
#
##
# TODO 5: Define calculates_results_stats function below, please be certain to replace None
#       in the return statement with the results_stats_dic dictionary that you create 
#       with this function
# 
def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the program run using classifier's model 
    architecture to classifying pet images. Then puts the results statistics in a 
    dictionary (results_stats_dic) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats_dic - Dictionary that contains the results statistics (either
                    a percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value. See comments above
                     and the previous topic Calculating Results in the class for details
                     on how to calculate the counts and statistics.
    """        
    # Creates empty dictionary for results_stats_dic
    results_stats_dic = dict()
    
    # Sets all counters to initial values of zero so that they can 
    # be incremented while processing through the images in results_dic 

    results_stats_dic['n_images'] = len(results_dic)          #number of images
    results_stats_dic['n_dogs_img'] = 0         # number of dog images
    results_stats_dic['n_match'] = 0            # number of matches between pet & classifier labels
    results_stats_dic['n_correct_dogs'] = 0     # number of correctly classified dog images
    results_stats_dic['n_correct_notdogs'] = 0  #- number of correctly classified NON-dog images']
    results_stats_dic['n_correct_breed'] = 0    # number of correctly classified dog breeds
#            n_images - number of images
#            n_dogs_img - number of dog images
#            n_notdogs_img - number of NON-dog images
#            n_match - number of matches between pet & classifier labels
#            n_correct_dogs - number of correctly classified dog images
#            n_correct_notdogs - number of correctly classified NON-dog images
#            n_correct_breed - number of correctly classified dog breeds

    #pet_label = list_val[1]
    #cls_label_l = [c.strip() for c in list_val[1].split(',')] 
    #if pet_label in cls_label_l:
    #    results_stats_dic['n_match'] +=1 
        # Interates through results_dic dictionary to recompute the statistics
        # outside of the calculates_results_stats() function
    for list_val in results_dic.values():

        # match (if dog then breed match)
        if list_val[2] == 1:
            results_stats_dic['n_match'] +=1

            # isa dog (pet label) & breed match
            if list_val[3] == 1:
                results_stats_dic['n_dogs_img'] +=1
                results_stats_dic['n_correct_breed'] +=1

                # isa dog (classifier label) & breed match
                if list_val[4] == 1:
                    results_stats_dic['n_correct_dogs'] +=1
            # NOT dog (pet_label)
            else:

                # NOT dog (classifier label)
                if list_val[4] == 0:
                    results_stats_dic['n_correct_notdogs'] +=1
        # NOT - match (not a breed match if a dog)
        else:

            # NOT - match
            # isa dog (pet label) 
            if list_val[3] == 1:
                results_stats_dic['n_dogs_img'] +=1 

                # isa dog (classifier label)
                if list_val[4] == 1:
                    results_stats_dic['n_correct_dogs'] +=1                    
            # NOT dog (pet_label)
            else:

                # NOT dog (classifier label)
                if list_val[4] == 0:
                    results_stats_dic['n_correct_notdogs'] +=1

    results_stats_dic['n_notdogs_img']  = results_stats_dic['n_images'] - results_stats_dic['n_dogs_img']
    #print("results_stats_dic['n_correct_breed'] = {}".format(results_stats_dic['n_correct_breed']))
    #print("results_stats_dic['n_correct_dogs'] = {}".format(results_stats_dic['n_correct_dogs']))


    # calculates number of not-a-dog images using - images & dog images counts
    #results_stats_dic['n_correct_notdogs'] =  results_stats_dic['n_images'] - results_stats_dic['n_dogs_img']  # number of correctly classified NON-dog images

    # TODO: 5c. REPLACE zero(0.0) with CODE that calculates the % of correctly
    #           matched images. Recall that this can be calculated by the
    #           number of correctly matched images ('n_match') divided by the 
    #           number of images('n_images'). This result will need to be 
    #           multiplied by 100.0 to provide the percentage.
    #    
    # Calculates % correct for matches
    results_stats_dic['pct_match'] =  (results_stats_dic['n_match'] / results_stats_dic['n_images']) * 100           # percentage of correct matches

    # TODO: 5d. REPLACE zero(0.0) with CODE that calculates the % of correctly
    #           classified dog images. Recall that this can be calculated by 
    #           the number of correctly classified dog images('n_correct_dogs')
    #           divided by the number of dog images('n_dogs_img'). This result 
    #           will need to be multiplied by 100.0 to provide the percentage.
    #    
    # Calculates % correct dogs
    results_stats_dic['pct_correct_dogs'] = (results_stats_dic['n_correct_dogs'] / results_stats_dic['n_dogs_img'] ) *100
 
    # TODO: 5e. REPLACE zero(0.0) with CODE that calculates the % of correctly
    #           classified breeds of dogs. Recall that this can be calculated 
    #           by the number of correctly classified breeds of dog('n_correct_breed') 
    #           divided by the number of dog images('n_dogs_img'). This result 
    #           will need to be multiplied by 100.0 to provide the percentage.
    #    
    # Calculates % correct breed of dog
    results_stats_dic['pct_correct_breed'] = (results_stats_dic['n_correct_breed'] / results_stats_dic['n_dogs_img']) * 100   # percentage of correctly classified dog breeds

    # Calculates % correct not-a-dog images
    # Uses conditional statement for when no 'not a dog' images were submitted 
    if results_stats_dic['n_notdogs_img'] > 0:
        results_stats_dic['pct_correct_notdogs'] = (results_stats_dic['n_correct_notdogs'] /
                                                results_stats_dic['n_notdogs_img'])*100.0
    else:
        results_stats_dic['pct_correct_notdogs'] = 0.0

        
    # TODO 5f. REPLACE None with the results_stats_dic dictionary that you 
    # created with this function 
    return results_stats_dic