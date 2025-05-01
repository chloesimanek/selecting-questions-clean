import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path

"""
Functions used to simulate student responses using IRT.

Imported and used in simulate_responses.ipynb.

Code by Tori Shen, Nathaniel Li, and Anna Rafferty
"""



def generate_student_skills(num_student, skill_min, skill_max, distribution,
                            num_skill=None, skill_list=None):
    '''
    generate a DataFrame where each row represents a student with skill scores
    distribution: Type of distribution ('uniform', 'normal', 'fat_tails')
    One of num_skill or skill list must be non-none
    '''
    if skill_list is not None:
        assert(num_skill is None or num_skill == len(skill_list))
        num_skill = len(skill_list)
    elif num_skill is not None:
        skill_list = [i+1 for i in range(num_skill)]
    else:
        raise Exception("One of num_skill or skill_list must be non-None!")

    if distribution == 'uniform':
        # Generate uniformly distributed scores
        scores = np.random.uniform(skill_min, skill_max, size=(num_student, num_skill))
    elif distribution == 'normal':
        # Generate normally distributed scores centered at the middle of the skill range
        mean = (skill_max + skill_min) / 2
        std_dev = (skill_max - skill_min) / 6  # Approx 99.7% data within range
        scores = np.random.normal(mean, std_dev, size=(num_student, num_skill))
        # Clip the scores to stay within the skill range
        scores = np.clip(scores, skill_min, skill_max)
    elif distribution == 'fat_tails':
        # Custom fat-tail distribution with peaks around 0.1 and 0.9
        def custom_fat_tail_distribution(size):
            # Beta distribution peaking near 0.1
            dist1 = np.random.beta(2, 8, size)
            # Beta distribution peaking near 0.9
            dist2 = np.random.beta(8, 2, size)
            # Combine the two distributions with 50% probability for each
            mix = np.where(np.random.random(size) < 0.5, dist1, dist2)
            return mix
        # Generate scores using the custom fat-tail distribution
        scores = custom_fat_tail_distribution((num_student, num_skill))
        # Scale scores to the skill range
        scores = skill_min + scores * (skill_max - skill_min)
    else:
        raise ValueError("Unknown distribution type. Choose 'uniform', 'normal', or 'fat_tails'.")

    # Create DataFrame with student IDs and skill scores
    # Skill columns are named with numbers from 1 to num_skill as strings
    df = pd.DataFrame(scores, columns=[str(skill) for skill in skill_list])
    df.insert(0, 'Student_ID', np.arange(num_student))
    return df

def generate_student_question_p_correct(df_student, df_CHAT, guess, question_per_student, discrimination):
    """
    Generate a DataFrame with student-question combinations and p_correct values.

    Parameters:
    - df_student: DataFrame with student IDs and skill scores.
    - df_CHAT: DataFrame with Itemid, skill, and difficulty.
    - guess: Guessing probability.
    - question_per_student: Number of questions each student answers.
    - discrimination: Discrimination parameter for the p_correct formula.

    Returns:
    - df_p_correct: DataFrame with columns 'studentid', 'itemid', and 'p_correct'.
    """
    num_student = df_student['Student_ID'].nunique()
    students = df_student['Student_ID'].values

    # Initialize list to collect data
    data = []

    # For each student
    df_student_new_index = df_student.set_index('Student_ID')
    for student_id in students:
        # Randomly select question_per_student questions for the student
        sampled_items = df_CHAT.sample(n=question_per_student, replace=False)
        cur_skills = df_student_new_index.loc[student_id][sampled_items['skill'].astype(str)]
        difficulty_skill_diff = sampled_items['difficulty'].values - cur_skills.values
        all_p_correct = guess + (1-guess) / (1 + np.exp(discrimination*(difficulty_skill_diff)))
        cur_df = pd.DataFrame(sampled_items[['skill', 'itemid']].reset_index(drop=True))
        cur_df['studentid'] = student_id
        cur_df['student_skill_level'] = cur_skills.values
        cur_df['item_difficulty'] = sampled_items['difficulty'].values
        cur_df['p_correct'] = all_p_correct
        data.append(cur_df)
        # for index, row in sampled_items.iterrows():
        #     itemid = row['itemid']
        #     skill = str(row['skill'])
        #     difficulty = row['difficulty']

        #     # # Get the student's score on the skill
        #     # score = df_student.loc[df_student['Student_ID'] == student_id, skill].values[0]

        #     # # Calculate p_correct
        #     # p_correct = guess + (1 - guess) / (1 + np.exp(discrimination * (difficulty - score)))

        #     # Append to data
        #     data.append({'studentid': student_id, 'skill': int(skill), 'itemid': itemid, 'p_correct': all_p_correct[index]})

    # Create DataFrame from data
    df_p_correct = pd.concat(data, ignore_index=True)#pd.DataFrame(data)
    return df_p_correct



#get correctness and shape results into responses df
def finalize_result(df_p_correct, num_student):

    df_p_correct['correct'] = np.random.binomial(1, df_p_correct['p_correct'])
    df_p_correct['item_order'] = df_p_correct.groupby('studentid').cumcount()
    df_p_correct['timeid'] = 0

    # Assign 'student_train_validate_test' to students
    num_student = df_p_correct['studentid'].nunique()
    students = df_p_correct['studentid'].unique()
    np.random.shuffle(students) 
    num_test = int(0.10 * num_student)
    num_validate = int(0.10 * num_student)
    num_train = num_student - num_test - num_validate
    student_labels = np.array([0]*num_train + [1]*num_validate + [2]*num_test)
    np.random.shuffle(student_labels)  # Shuffle labels to randomize assignment
    student_label_mapping = dict(zip(students, student_labels))
    df_p_correct['student_train_validate_test'] = df_p_correct['studentid'].map(student_label_mapping)

    # Reorder columns
    df_final = df_p_correct[['studentid', 'itemid', 'skill', 'item_difficulty', 'student_skill_level', 
                             'p_correct', 'correct', 'item_order', 'timeid', 'student_train_validate_test']]

    return df_final

def simulate_responses(num_student, question_per_student, question_df, skill_distribution, 
                       num_skill=None, skill_list=None):
    #make customizations here, and then run everything else
    #you can check out the outputs of the cells first, I used sample_chat_data.csv and the numbers below to yield them

    # the following customizations are set to default
    # can change these, but wouldn't make much such sense since these numbers affect each other
    discrimination=1
    skill_min = -4
    skill_max = 4
    guess=0.1

    if isinstance(question_df, str):
        df_CHAT = pd.read_csv(question_df)
    else:
        df_CHAT = question_df

    # Generate data with uniform distribution
    if skill_list is not None:
        assert(num_skill is None or num_skill == len(skill_list))
        df_student = generate_student_skills(num_student, skill_min, skill_max, skill_distribution, skill_list=skill_list)
    elif num_skill is not None:
        df_student = generate_student_skills(num_student, skill_min, skill_max, skill_distribution, num_skill=num_skill)
    else:
        raise Exception("One of num_skill or skill_list must be non-None!")
    # Generate the student-question p_correct DataFrame
    df_p_correct = generate_student_question_p_correct(df_student, df_CHAT, guess, question_per_student, discrimination)
    df_final = finalize_result(df_p_correct, num_student)

    return df_final, df_student
 
def load_eedi_df(metadata_file='eedi\question_metadata_task_3_4.csv',  
                 questions_file='eedi\question_embeddings.csv',
                 embeddings_file='eedi\mathbert_embeddings.csv'):
    '''
    Load a df with all questions from the eedi data for which we have question embeddings,
    and generate random difficulties for each question. Return the df, which will also
    include skill alignments and the raw text of the questions.
    '''
    metadata = pd.read_csv(metadata_file)
    questions = pd.read_csv(questions_file)
    df = pd.merge(metadata, questions, on='QuestionId')
    items_with_bert_embeddings = pd.read_csv(embeddings_file)['itemid'].values
    df = df[df['QuestionId'].isin(items_with_bert_embeddings)]

    def clean_text(text):
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        while '  ' in text:
            text = text.replace('  ', ' ')

        return text

    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].apply(clean_text)
    df['SubjectId'] = df['SubjectId'].apply(lambda x: ast.literal_eval(x))

    # Setting the 1D skill used for reponse generation to be based on the third level of the subject heirarchy as a good compromise
    # between specificity and generalization
    df['skill'] = df['SubjectId'].apply(lambda x: x[2])
    df['difficulty'] = np.random.normal(0, 1,size=len(df)) #np.random.randint(-2, 3, size=len(df))

    df.rename(columns={'QuestionId': 'itemid'}, inplace=True)
    df.rename(columns={'SubjectId': 'subject_taxonomy'}, inplace=True)

    skills = df['skill'].unique()
    mapping = {skill: i + 1 for i, skill in enumerate(skills)}
    df['skill'] = df['skill'].apply(lambda x: mapping[x])
    return df

def filter_skills(df, min_questions_per_skill=20, max_questions_per_skill=110,
                  max_overall_skills=None):
    '''
    Return a df containing only questions aligned to skills where the skill has at
    least min_questions_per_skill aligned to it and at most max_questions_per_skill
    aligned to it. If max_overall_skills is not None, this further subsamples to only include
    a maximum of max_overall_skills skills in the final df.
    '''
    counts = df['skill'].value_counts()
    skills_to_include = counts[(counts >= min_questions_per_skill) & (counts <= max_questions_per_skill)].index
    print(skills_to_include)

    if max_overall_skills is not None:
        skills_to_include = np.random.choice(skills_to_include, 5, replace=False)
    df = df[df['skill'].isin(skills_to_include)]
    return df

def sample_items_by_skill(df, num_items_per_skill):
    '''
    Returns a dataframe containing exactly num_items_per_skill items for each skill in df.
    df is assumed to have at least num_items_per_skill items for each skill.
    '''
    return df.sample(frac=1).reset_index(drop=True).groupby(by=['skill']).head(num_items_per_skill).reset_index(drop=True)


def eedi_simulation(num_students=50000, min_questions_per_skill=20, max_questions_per_skill=110, max_overall_skills=5):
    df = load_eedi_df()

    print("Number of unique skills: ", df['skill'].nunique())
    
    simulate_and_save(df, 'responses',
                      num_students=50000, 
                      min_questions_per_skill=20, 
                      max_questions_per_skill=110, 
                      max_overall_skills=5,
                      questions_per_student=100,
                      skill_distribution='normal')


def simulate_and_save(question_df, output_directory, 
                      num_students=50000, 
                      min_questions_per_skill=20, 
                      max_questions_per_skill=120, 
                      max_overall_skills=5,
                      questions_per_student=100,
                      skill_distribution='normal'):
    df = filter_skills(question_df, min_questions_per_skill=min_questions_per_skill, 
                       max_questions_per_skill=max_questions_per_skill, 
                       max_overall_skills=max_overall_skills)
    skills_to_include = df['skill'].unique()
     
    df_responses, df_student = simulate_responses(num_students, questions_per_student, df, skill_distribution, skill_list=skills_to_include)
    Path(f'{output_directory}').mkdir(parents=True, exist_ok=True)

    df_responses.to_csv(f'{output_directory}/responses_df.csv', index=False)
    df_student.to_csv(f'{output_directory}/student_df.csv', index=False)

def gpt_simulation(num_students=50000, min_questions_per_skill=20, max_questions_per_skill=120, max_overall_skills=5):
    df = pd.read_csv('../data/GPT/questions_with_embeddings_actual.csv')

    print("Number of unique skills: ", df['skill'].nunique())
    
    simulate_and_save(df, '../simulated_responses/gpt_randn/',
                      num_students=50000, 
                      min_questions_per_skill=20, 
                      max_questions_per_skill=120, 
                      max_overall_skills=5,
                      questions_per_student=100,
                      skill_distribution='normal')

def eedi_oracle_simulation(num_students=500, min_questions_per_skill=20, max_questions_per_skill=110, max_overall_skills=5):
    
    df = load_eedi_df()
    print("Number of unique skills: ", df['skill'].nunique())
    
    simulate_oracle_and_save(df, 'responses',
                      num_students=500, 
                      min_questions_per_skill=20, 
                      max_questions_per_skill=110, 
                      max_overall_skills=5,
                      questions_per_student=100,
                      skill_distribution='normal')

def simulate_oracle_and_save(question_df, output_directory, 
                             num_students=500, 
                             min_questions_per_skill=20, 
                             max_questions_per_skill=110, 
                             max_overall_skills=5,
                             questions_per_student=100,
                             skill_distribution='normal'):
    
    df = filter_skills(question_df, min_questions_per_skill=min_questions_per_skill, 
                       max_questions_per_skill=max_questions_per_skill, 
                       max_overall_skills=max_overall_skills)
    skills_to_include = df['skill'].unique()
    
    df_responses, df_student = simulate_responses(num_students, questions_per_student, df, skill_distribution, skill_list=skills_to_include)
    Path(f'{output_directory}').mkdir(parents=True, exist_ok=True)

    save_clean_oracle_data(df_responses, f'{output_directory}/oracle_dataset.csv')

    df_responses.to_csv(f'{output_directory}/oracle_responses_df.csv', index=False)
    df_student.to_csv(f'{output_directory}/oracle_student_df.csv', index=False)

# Filter for the columns you need
def save_clean_oracle_data(df_responses, output_file='oracle_dataset.csv'):
    df_clean = df_responses[['studentid', 'itemid', 'item_difficulty', 'student_skill_level', 'correct']]
    df_clean.to_csv(output_file, index=False)
    print(f"Saved clean oracle dataset to: {output_file}")


def main():
    eedi_oracle_simulation()
    # gpt_simulation()

if __name__ == "__main__":
    main()