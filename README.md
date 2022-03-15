### Music-Matcher
 
This code exists to identify pairs of audio files containing the same audio, although possibly encoded or compressed with different standards/filetypes.

Identical audio is identified by (1) matching duration and (2) high similarity as measured through dynamic time warping. The filenames are irrelevant.

More practically, the code solves two problems pertaining to the management of music libraries:

Use Case 1: I have an old library of low-quality music file downloads (set A). More recently, I downloaded higher-quality versions of some of these files (set B). Some songs exist in both sets, some songs exist in only set A, and some songs exist in only set B. I wish to merge these two libraries together.

Use Case 2. I have, on several occasions, accidently downloaded multiple copies of the same song, resulting in duplicates in my library. I want to remove these duplicates, but it would be too time consuming to find them manually.

# Workflow for Use Case 1:

1. Place the music in set A into one directory, and the music in set B into another. Also create a directory for the script to save output files.
2. Modify lines 16-18 in `src/main.py`, as well as lines 13-15 in `src/intersection_and_union.py`, to match these directories.
3. Run `src/main.py`. No arguments are necessary. This will take a long time if you have a large (hundreds or thousands of files) library.
4. Copy `OUTPUT_DIRECTORY/matches.csv` to `OUTPUT_DIRECTORY/matches_modified.csv`.
5. Inspect `OUTPUT_DIRECTORY/matches_modified.csv`. Each row represents a match between set A and set B. Matches are sorted by decreasing confidence, which means erroneous matches will be located at the bottom of the file. Delete rows containing these erroneous matches.
6. Run `src/intersection_and_union.py`. This will generate four subdirectories in the output directory. `OUTPUT_DIRECTORY/dir_a_matched/` and `OUTPUT_DIRECTORY/dir_b_matched` contain the audio files which were successfully matched; you can delete whichever directory contains inferior-quality files. `OUTPUT_DIRECTORY/dir_a_unmatched` and `OUTPUT_DIRECTORY/dir_b_unmatched` contain audio which was found to exist only in set A or only in set B, respectively.
7. As the algorithm is not perfect, there may be a very small number (~0.5% in my experience) of cases where two files containing identical audio fail to be matched. Depending on how much you care about avoiding duplicate audio files, you may wish to manually check for these cases.

# Workflow for Use Case 2:

WIP

# TODO:

- workflow for use case 2
- further improve robustness to differences in volume
