

llm_prompt = """Evaluate the given solution (not necessarily complete) based on the clarity, coherence, and correctness of the logical reasoning process. Assess whether the steps follow a structured, step-by-step approach, ensuring that intermediate steps are neither skipped nor incorrect. Consider whether assumptions are clearly stated, conclusions logically follow from premises, and the response adheres to formal rules in mathematical proofs or coding logic. 

Provide only decimal score between 0 and 10. The output format is limited to: "Score:..." where ... represents the omitted output content, which you need to fill in.
Here is the input you should score.
Input: 
Problem:"""

image_description_score = '''Your task is to evaluate how well the image description describes the given image. The score should be a decimal between 0 and 10.  Higher accuracy of description should result in a higher score. 
Your scoring should be entirely based on the input image description and image provided. Do not generate additional Image Description or solution.
The output format is limited to: "Score:...", where ... represents the omitted output content, which you need to fill in.
Here is the image description, Only generate the decimal score and follow the restricted format. Do not generate additional analysis or explanation.
Input:
'''
zero_single_proposal_prompt_en = '''
Given an image, image description, and an existing partial solution (not a complete answer), generate the correct next step to solve the problem.
Please keep your response concise and limited to one reasoning step.

The output format is strictly:
"Next step: ..."
Please follow the restricted format and be brief.
'''

# critic_simplified = '''Given problem, image, image description, evaluate whether the reasoning steps can solve the provided problem and output a score. The score should be a decimal between 0 and 10. If all the steps are correct and the answer is calculated, the score is 10. The closer the steps are to the final answer, the closer the score is to 10. If there is no correct answer, never give a score higher than 9."
# Now, given a problem, image, image description and the provided steps, provide only the decimal score entirely based on the input image, image description,reasoning steps provided.
# The output format is limited to: "Score:...", where ... represents the omitted decimal score, which you need to fill in.
# Follow the output format.
# '''
# critic_simplified = '''Given problem, image, image description, give the score between 0 and 10 whether the reasoning steps can solve the problem. If there is no correct answer, never give a score higher than 9."
# The output format is limited to: "Score:...", where ... represents the omitted decimal score, which you need to fill in. Do not generate additional reasoning step or anaylsis. Follow the output format.'''
critic_simplified = '''Given the problem, image, image description, and reasoning steps, evaluate whether the reasoning steps are sufficient to solve the problem.
If there is no correct answer, never assign a score higher than 9.
Output format must be: "Score: ...", where ... is the decimal score.
Do not include any additional explanation or analysis. Follow the output format exactly.'''


single_reflection_prompt_simple_en = '''
Given a problem, image, image description and reasoning steps (not necessarily complete), you need to determine whether the given steps have completely solved the problem.
If the given steps have provided the final answer to the question, then you should generate "Problem solved" and nothing else.
If the given steps have not yet calculated the answer to the question or have not finished reasoning, then please generate "Problem unsolved" 
Do not generate additional reasoning step or anaylsis, please generate either "Problem solved" or "Problem unsolved". 
'''


# image_description_prompt = '''
# your task is to provide the image description based on a given image and problem, the output format is limited to "Image Description:..." where ... represents the output result, which you should fill in.
# Here is the input, please follow the restricted output format.
# Given problem:
# '''

image_description_prompt = "Briefly describe the image in a few sentences."
# "Generate a very simple, short and accurate image description based on the given image."

# zero_single_proposal_prompt_en = '''
# Given image, image description and an existing partial solution (not a complete answer), generate the correct next step to solve the problem
# The output format is limited to:
# "Next step: ..."
# where ... indicates the part you should fill in. Your output should be a only one reasoning step to solve the problem
# Here is the input, please follow the restricted output format. 

# Problem: '''

