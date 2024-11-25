# 2024-2025 M2 QDCS GPU Programming

This course focuses on developing efficient parallel programs to run on GPU architectures. We will specifically focus on using the CUDA language and Nvidia GPUs. Many GPU architecture details and various code optimization techniques will be covered throughout the course.

# Course organization

# Evaluation

## Graded lab assignments
Each lab assignment should be sent to the address `oguz.kaya[at]universite-paris-saclay.fr` with the subject format **"M2QDCSGPU LABX SURNAME(s) Name(s)"** (e.g., **M2QDCSGPU LAB3 ARNAULT-BARRAT Jean-François**) by 23:59 Sunday following the lab session. Solutions will be posted on the git repository just after this deadline; therefore, do not submit late! Please follow this format for the email subject **to the letter** as I will employ filters to sort these submissions.

Please only attach source files (*.cpp) to your email, **one file per exercise** (e.g., if the assignment has four exercises, your submission must have four .cpp files), and please **do not zip the files**!

If there are plots in some exercises, **do not** include them in the submission; those are only for your better understanding of the code's behavior, not for evaluation. If there are other questions/interpretations requiring textual response as part of an exercise, you can put your  at the very beginning of the corresponding source file (*.cpp) in a comment section.

Out of all lab assignments, two of them will be randomly selected for grading, each providing up to 2.5/20 points.

## End-of-session mini-exams
At the end of weeks 4, 5, 6 there will be mini-exams for 5/20 points each. Three types of exercises that might appear on these mini-exams are:

**Debugging:** Given a complete code with multiple bugs, the goal is to identify each bug (without correcting them), say what the problem is in a single sentence, and point out the corresponding line(s) in the code.

**Output:** Given a complete code, the goal is to find out what the program could potentially display. Indeed, this requires a thorough understanding of how the code works. There might be multiple possibilities, in which case you should indicate all of them.

**Coding:** A squeleton code or a function signature will be given for a specific task, and you will be expected to provide a parallel/optimized code that performs the intended task.

You can bring four A4 sheets to the mini-exam (both sides); no other material is allowed.

# Technical tips

Text editor: If you do not have a "preferred" text editor for development already, I recommend Visual Studio Code: https://code.visualstudio.com . You can follow the official tutorial for the basics: https://www.youtube.com/watch?v=B-s71n0dHUk&list=PLj6YeMhvp2S5UgiQnBfvD7XgOMKs3O_G6

If your nvcc command does not work on the reserved machines, put the following line to your ~/.bashrc file:

PATH=/usr/local/cuda/bin:$PATH

## Contact
  oguz.kaya [at] universite-paris-saclay.fr
