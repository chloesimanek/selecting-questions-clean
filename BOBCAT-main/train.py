import numpy as np
import torch
import sys
import os
from dataset import Dataset, collate_fn
from utils.utils import compute_auc, compute_accuracy, data_split, batch_accuracy
from model import MAMLModel
from policy import PPO, Memory, StraightThrough
from copy import deepcopy
from utils.configuration import create_parser, initialize_seeds
import time
import os
import json
DEBUG = False if torch.cuda.is_available() else True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_score, best_test_score = 0, 0
best_val_auc, best_test_auc = 0, 0
best_epoch = -1

# LOG CODE: Added flag to control detailed logging of question selection process
PRINT_QUESTION_SELECTION = False

# Store the raw data for reference to original user_ids and q_ids
raw_train_data = None

# Function to get actual student ID from batch index
def get_actual_student_id(batch_idx, student_idx):
    """Get the actual student ID from raw data if available"""
    batch_offset = batch_idx * params.train_batch_size
    raw_idx = batch_offset + student_idx
    if raw_train_data and raw_idx < len(raw_train_data):
        return raw_train_data[raw_idx].get('user_id', f"batch-{batch_idx}-student-{student_idx}")
    return f"batch-{batch_idx}-student-{student_idx}"

def clone_meta_params(batch):
    return [meta_params[0].expand(len(batch['input_labels']),  -1).clone()]


def inner_algo(batch, config, new_params, create_graph=False):
    for _ in range(params.inner_loop):
        config['meta_param'] = new_params[0]
        res = model(batch, config)
        loss = res['train_loss']
        grads = torch.autograd.grad(
            loss, new_params, create_graph=create_graph)
        new_params = [(new_params[i] - params.inner_lr*grads[i])
                      for i in range(len(new_params))]
        del grads
    config['meta_param'] = new_params[0]
    return


def get_rl_baseline(batch, config):
    model.pick_sample('random', config)
    new_params = clone_meta_params(batch)
    inner_algo(batch, config, new_params)
    with torch.no_grad():
        output = model(batch, config)['output']
    random_baseline = batch_accuracy(output, batch)
    return random_baseline


# LOG CODE: Added detailed logging in this function to visualize RL policy decisions
def pick_rl_samples(batch, config):
    """Select questions using reinforcement learning policy"""
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    
    # LOG CODE: Print student parameters and available questions
    if PRINT_QUESTION_SELECTION and isinstance(memory.states, list) and len(memory.states) == 0:
        print("\n--- Question Selection (RL Policy) ---")
        n_students = len(batch['input_labels'])
        for student_idx in range(2):  # Print for at most 2 students to avoid clutter
            # Use detach() to avoid the "requires grad" error
            student_param = config['meta_param'][student_idx].detach().cpu().numpy()
            
            # Get actual student ID
            actual_student_id = get_actual_student_id(batch_idx, student_idx)
            
            print(f"Student {student_idx+1} (ID: {actual_student_id}) initial parameter (theta): {student_param}")
            
            # Show available questions
            available = torch.where(action_mask[student_idx] > 0)[0].cpu().numpy()
            print(f"Student {student_idx+1} has {len(available)} available questions")
    
    # Select questions one by one
    for q_num in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
        
        if config['mode'] == 'train':
            actions = ppo_policy.policy_old.act(state, memory, action_mask)
        else:
            with torch.no_grad():
                actions = ppo_policy.policy_old.act(state, memory, action_mask)
        
        # LOG CODE: Log the selected questions
        if PRINT_QUESTION_SELECTION and q_num == 0:  # Only print for the first question to avoid clutter
            print("\nSelected Questions:")
            print("Student | Question | Actual-IDs: User-Question | Current Param")
            print("-" * 65)
            for student_idx, q_id in enumerate(actions[:min(5, len(actions))].cpu().numpy()):
                # Use detach() to avoid the "requires grad" error
                student_param = config['meta_param'][student_idx].detach().cpu().numpy()
                
                # Get actual student ID
                actual_student_id = get_actual_student_id(batch_idx, student_idx)
                
                print(f"{student_idx+1:7d} | {q_id:8d} | {actual_student_id}-{q_id:8d} | {student_param}")
        
        # Update masks
        action_mask[range(len(action_mask)), actions], train_mask[range(
            len(train_mask)), actions] = 0, 1
        env_states['train_mask'], env_states['action_mask'] = train_mask, action_mask
    
    # Store final training mask
    config['train_mask'] = env_states['train_mask']
    return


def run_unbiased(batch, config):
    new_params = clone_meta_params(batch)
    config['available_mask'] = batch['input_mask'].to(device).clone()
    # LOG CODE: Added this line to avoid KeyError
    config['meta_param'] = new_params[0]
    if config['mode'] == 'train':
        random_baseline = get_rl_baseline(batch, config)
    pick_rl_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
        final_accuracy = batch_accuracy(res['output'], batch)
        reward = final_accuracy - random_baseline
        memory.rewards.append(reward.to(device))
        ppo_policy.update(memory)
        #
    else:
        with torch.no_grad():
            res = model(batch, config)
    memory.clear_memory()
    return res['output']


# LOG CODE: Enhanced this function with detailed logging
def pick_biased_samples(batch, config):
    """Select questions using the biased strategy"""
    new_params = clone_meta_params(batch)
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    
    # LOG CODE: Print student parameters
    if PRINT_QUESTION_SELECTION:
        print("\n--- Question Selection (Biased Policy) ---")
        n_students = len(batch['input_labels'])
        for student_idx in range(2):  # Print for at most 2 students to avoid clutter
            # Use detach() to avoid the "requires grad" error
            student_param = config['meta_param'][student_idx].detach().cpu().numpy()
            
            # Get actual student ID
            actual_student_id = get_actual_student_id(batch_idx, student_idx)
            
            print(f"Student {student_idx+1} (ID: {actual_student_id}) initial parameter (theta): {student_param}")
    
    for q_num in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
            train_mask = env_states['train_mask']
        
        if config['mode'] == 'train':
            train_mask_sample, actions = st_policy.policy(state, action_mask)
        else:
            with torch.no_grad():
                train_mask_sample, actions = st_policy.policy(
                    state, action_mask)
        
        # LOG CODE: Log the selected questions
        if PRINT_QUESTION_SELECTION and q_num == 0:  # Only print for the first question to avoid clutter
            print("\nSelected Questions:")
            print("Student | Question | Actual-IDs: User-Question | Current Param")
            print("-" * 65)
            for student_idx, q_id in enumerate(actions[:min(5, len(actions))].cpu().numpy()):
                # Use detach() to avoid the "requires grad" error
                student_param = config['meta_param'][student_idx].detach().cpu().numpy()
                
                # Get actual student ID
                actual_student_id = get_actual_student_id(batch_idx, student_idx)
                
                print(f"{student_idx+1:7d} | {q_id:8d} | {actual_student_id}-{q_id:8d} | {student_param}")
        
        action_mask[range(len(action_mask)), actions] = 0
        # env state train mask should be detached
        env_states['train_mask'], env_states['action_mask'] = train_mask + \
            train_mask_sample.data, action_mask
        if config['mode'] == 'train':
            # loss computation train mask should flow gradient
            config['train_mask'] = train_mask_sample+train_mask
            inner_algo(batch, config, new_params, create_graph=True)
            res = model(batch, config)
            loss = res['loss']
            st_policy.update(loss)
    
    config['train_mask'] = env_states['train_mask']
    return


def run_biased(batch, config):
    new_params = clone_meta_params(batch)
    # LOG CODE: Added this line to avoid KeyError
    config['meta_param'] = new_params[0]
    if config['mode'] == 'train':
        model.eval()
    pick_biased_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        model.train()
        optimizer.zero_grad()
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
    else:
        with torch.no_grad():
            res = model(batch, config)
    return res['output']


def run_random(batch, config):
    """Run training with random question selection strategy"""
    new_params = clone_meta_params(batch)
    # LOG CODE: Added this line to avoid KeyError
    config['meta_param'] = new_params[0]
    meta_params_optimizer.zero_grad()
    if config['mode'] == 'train':
        optimizer.zero_grad()
    ###
    config['available_mask'] = batch['input_mask'].to(device).clone()
    config['train_mask'] = torch.zeros(
        len(batch['input_mask']), params.n_question).long().to(device)

    # LOG CODE: Print student parameters and available questions
    if PRINT_QUESTION_SELECTION and config['mode'] == 'train':
        print("\n--- Question Selection (Random/Active) ---")
        n_students = len(batch['input_labels'])
        for student_idx in range(2):  # Print for at most 2 students to avoid clutter
            # Use detach() to avoid the "requires grad" error
            student_param = config['meta_param'][student_idx].detach().cpu().numpy()
            
            # Get actual student ID
            actual_student_id = get_actual_student_id(batch_idx, student_idx)
            
            print(f"Student {student_idx+1} (ID: {actual_student_id}) initial parameter (theta): {student_param}")
    
    # Random pick once

    if sampling == 'random':
        # config['diffs'] = batch['diffs'].to(device)
        # LOG CODE
        if PRINT_QUESTION_SELECTION and config['mode'] == 'train':
            config['diffs'] = batch['diffs'].to(device)

            # Get current train_mask before selection
            prev_mask = config['train_mask'].clone()
            model.pick_sample('random', config)
            # Find newly selected questions
            curr_mask = config['train_mask']
            n_students = len(batch['input_labels'])  # Define n_students here
            for student_idx in range(2):  # Print 2 students
                selected_qs = torch.where((curr_mask[student_idx] == 1) & (prev_mask[student_idx] == 0))[0].cpu().numpy()
                if len(selected_qs) > 0:
                    # Get actual student ID
                    actual_student_id = get_actual_student_id(batch_idx, student_idx)
                    
                    # Print selected questions with actual IDs
                    print(f"Student {student_idx+1} (ID: {actual_student_id}) random questions:")
                    for q in selected_qs[:5]:
                        print(f"  Question {q} (ID: {q})")
                    if len(selected_qs) > 5:
                        print(f"  ... and {len(selected_qs)-5} more")
        else:
            model.pick_sample('random', config)
        inner_algo(batch, config, new_params)
    
    # ORACLE 
    if sampling == 'oracle':
        # batch to config to select questions from 
        config['diffs'] = batch['diffs'].to(device)
        # LOG CODE 
        if PRINT_QUESTION_SELECTION and config['mode'] == 'train':
            # print("train")
            # Get current train_mask before selection
            prev_mask = config['train_mask'].clone()
            # PICK SAMPLE HERE -> pick_oracle_sample
            model.pick_sample('oracle', config)
            # model.pick_sample('random', config)
            # Find newly selected questions
            curr_mask = config['train_mask']
            diffs = torch.abs(config['diffs']) 
            
            input_mask = config['available_mask']
            n_students = len(batch['input_labels'])

            for student_idx in range(2):  # Print 2 students
                # WAS 0 -> NOW 1 (SELECTED)
                selected_qs = torch.where((curr_mask[student_idx] == 1) & (prev_mask[student_idx] == 0))[0].cpu().numpy() 
                if len(selected_qs) > 0:
                    # Get actual student ID
                    actual_student_id = get_actual_student_id(batch_idx, student_idx)

                    print(f"\nStudent {student_idx+1} (ID: {actual_student_id}) oracle questions:")
                    diffs_info = diffs[student_idx].detach().cpu()
                    input_mask_info = input_mask[student_idx].detach().cpu()
                    
                    # For each selected question, print ID and absolute diff
                    diffs_chosen = []
                    for q in selected_qs:
                        d = diffs_info[q].item()
                        diffs_chosen.append(d)
                        print(f"  Question {q} difference = {d:.4f})")

                    # Show min abs(diff) of all available questions 
                    available_diffs = torch.abs(diffs_info[input_mask_info == 1])
                    if len(available_diffs) > 0:
                        min_diff = available_diffs.min().item()
                        print(f"Minimum available difference: {min_diff:.4f}")

                    # Ensure min abs(diff) was chosen
                    if min_diff in diffs_chosen:
                        print('Chosen: YES')
                    else:
                        print('Chosen: NO')

        else:
            # print("test")
            model.pick_sample('oracle', config)

        inner_algo(batch, config, new_params)

        # config['diffs'] = batch['diffs'].to(device)
        # model.pick_sample('oracle', config)
        # inner_algo(batch, config, new_params)
    
    elif sampling == 'bad':
        # batch to config to select questions from
        config['diffs'] = batch['diffs'].to(device)

        # LOG CODE
        if PRINT_QUESTION_SELECTION and config['mode'] == 'train':
            prev_mask = config['train_mask'].clone()

            # PICK SAMPLE HERE -> pick_bad_sample
            model.pick_sample('bad', config)

            curr_mask = config['train_mask']
            diffs = torch.abs(config['diffs'])
            input_mask = config['available_mask']
            n_students = len(batch['input_labels'])

            for student_idx in range(2):  # Print 2 students
                selected_qs = torch.where((curr_mask[student_idx] == 1) & (prev_mask[student_idx] == 0))[0].cpu().numpy()
                if len(selected_qs) > 0:
                    actual_student_id = get_actual_student_id(batch_idx, student_idx)

                    print(f"\nStudent {student_idx+1} (ID: {actual_student_id}) BAD questions:")
                    diffs_info = diffs[student_idx].detach().cpu()
                    input_mask_info = input_mask[student_idx].detach().cpu()

                    diffs_chosen = []
                    for q in selected_qs:
                        d = diffs_info[q].item()
                        diffs_chosen.append(d)
                        print(f"  Question {q} difference = {d:.4f}")

                    # Show max abs(diff) of all available questions
                    available_diffs = torch.abs(diffs_info[input_mask_info == 1])
                    if len(available_diffs) > 0:
                        max_diff = available_diffs.max().item()
                        print(f"Maximum available difference: {max_diff:.4f}")

                    if max_diff in diffs_chosen:
                        print('Chosen: YES')
                    else:
                        print('Chosen: NO')
        else:
            model.pick_sample('bad', config)

        inner_algo(batch, config, new_params)


    # LOG CODE: Added extensive logging for active selection
    if sampling == 'active':
        if PRINT_QUESTION_SELECTION and config['mode'] == 'train':
            print("\nSelecting questions using active strategy:")
            n_students = len(batch['input_labels'])
            
            for q_idx in range(params.n_query):
                # Get current available and training masks
                available_mask = config['available_mask'].clone()
                
                # Get model output for uncertainty score calculation
                with torch.no_grad():
                    output = model.compute_output(config['meta_param']) # Gets model prediction for student-question pair
                    output = torch.sigmoid(output)
                    
                    # Calculate uncertainty scores (same as in model.py)
                    inf_mask = torch.clamp(torch.log(available_mask.float()), 
                                          min=torch.finfo(torch.float32).min)
                    scores = torch.min(1-output, output) + inf_mask
                
                # Print top uncertain questions for a few students
                for student_idx in range(2):
                    # Use detach() to avoid the "requires grad" error
                    student_param = config['meta_param'][student_idx].detach().cpu().numpy()
                    
                    # Get actual student ID
                    actual_student_id = get_actual_student_id(batch_idx, student_idx)
                    
                    # Get available questions and scores
                    available_qs = torch.where(available_mask[student_idx] > 0)[0]
                    if len(available_qs) == 0:
                        continue
                        
                    q_scores = scores[student_idx, available_qs]
                    
                    # Get top 5 questions
                    top_indices = torch.argsort(q_scores, descending=True)[:min(5, len(q_scores))]
                    top_questions = available_qs[top_indices].cpu().numpy()
                    top_scores = q_scores[top_indices].cpu().numpy()
                    
                    # Extract question difficulties if available
                    if hasattr(model, 'question_difficulty'): # only works in IRT case
                        with torch.no_grad():
                            difficulties = model.question_difficulty.detach().cpu().numpy()
                            if difficulties.ndim > 1 and difficulties.shape[0] == 1:
                                difficulties = difficulties[0]
                            difficulties_str = [f"{difficulties[q]:.4f}" for q in top_questions]
                    else:
                        difficulties_str = ["N/A" for q in top_questions]
                    
                    print(f"\nStudent {student_idx+1} (ID: {actual_student_id}, param={student_param}) question {q_idx+1} candidates:")
                    for q, score, diff in zip(top_questions, top_scores, difficulties_str):
                        print(f"  Question {q} (ID: {q}): uncertainty={score:.4f}, difficulty={diff}")
                
                # Select and print the actual questions
                old_available = config['available_mask'].clone()
                model.pick_sample('active', config)
                
                for student_idx in range(2):
                    # Find what was selected by comparing masks
                    old_avail = old_available[student_idx].cpu().numpy()
                    new_avail = config['available_mask'][student_idx].cpu().numpy()
                    selected_q = np.where((old_avail == 1) & (new_avail == 0))[0]
                    
                    if len(selected_q) > 0:
                        q_id = selected_q[0]
                        # Get actual student ID
                        actual_student_id = get_actual_student_id(batch_idx, student_idx)
                        
                        if hasattr(model, 'question_difficulty'):
                            with torch.no_grad():
                                difficulties = model.question_difficulty.detach().cpu().numpy()
                                if difficulties.ndim > 1 and difficulties.shape[0] == 1:
                                    difficulties = difficulties[0]
                                difficulty = difficulties[q_id]
                        else:
                            difficulty = "N/A"
                        print(f"Student {student_idx+1} (ID: {actual_student_id}) selected Q{q_idx+1}: {q_id} (difficulty={difficulty})")
                
                # Save student parameter before update
                with torch.no_grad():
                    old_param = config['meta_param'].clone().detach()
                
                # Update student parameters for visualization
                if q_idx < params.n_query - 1:  # Skip last update as it won't be used for selection
                    temp_params = clone_meta_params(batch)
                    inner_algo(batch, config, temp_params)
                    
                    # Print parameter update for a few students
                    if PRINT_QUESTION_SELECTION:
                        for student_idx in range(2):
                            old_p = old_param[student_idx].detach().cpu().numpy()
                            new_p = config['meta_param'][student_idx].detach().cpu().numpy()
                            # Get actual student ID
                            actual_student_id = get_actual_student_id(batch_idx, student_idx)
                            
                            print(f"Student {student_idx+1} (ID: {actual_student_id}) parameter updated: {old_p} -> {new_p}")
        else:
            for _ in range(params.n_query):
                model.pick_sample('active', config)
                # Need to update student parameters after each question
                inner_algo(batch, config, new_params)

    if config['mode'] == 'train':
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        return
    else:
        with torch.no_grad():
            res = model(batch, config)
        output = res['output']
        return output


def train_model():
    global best_val_auc, best_test_auc, best_val_score, best_test_score, best_epoch, batch_idx
    config['mode'] = 'train'
    config['epoch'] = epoch
    model.train()
    N = [idx for idx in range(100, 100+params.repeat)]
    
    # LOG CODE: Print training epoch information
    print(f"\n===== EPOCH {epoch} =====")
    
    # LOG CODE: Calculate average student parameters
    with torch.no_grad():
        avg_theta = meta_params[0].mean().cpu().numpy()
        print(f"Average student parameter (theta): {avg_theta:.4f}")
    
    # LOG CODE: If model has question difficulty, print some statistics
    if hasattr(model, 'question_difficulty'):
        with torch.no_grad():
            difficulties = model.question_difficulty.cpu().numpy()
            if difficulties.ndim > 1 and difficulties.shape[0] == 1:
                difficulties = difficulties[0]
            print(f"Question difficulty stats: min={difficulties.min():.4f}, mean={difficulties.mean():.4f}, max={difficulties.max():.4f}")
    
    for batch_idx, batch in enumerate(train_loader):
        # LOG CODE: For logging purposes, only print detailed selection for a few batches
        global PRINT_QUESTION_SELECTION
        PRINT_QUESTION_SELECTION = (batch_idx < 1)  # Only print for first batch - CHANGE IF NEEDED
        
        if PRINT_QUESTION_SELECTION:
            print(f"\nBatch {batch_idx+1} ({len(batch['input_labels'])} students)")
        
        # Select RL Actions, save in config
        if sampling == 'unbiased':
            run_unbiased(batch, config)
        elif sampling == 'biased':
            run_biased(batch, config)
        else:
            run_random(batch, config)
        
        # LOG CODE: Turn off printing after first few batches to avoid excessive output
        if batch_idx >= 1:
            PRINT_QUESTION_SELECTION = False
    
    # Validation
    val_scores, val_aucs = [], []
    test_scores, test_aucs = [], []
    for idx in N:
        _, auc, acc = test_model(id_=idx, split='val')
        val_scores.append(acc)
        val_aucs.append(auc)
    val_score = sum(val_scores)/(len(N)+1e-20)
    val_auc = sum(val_aucs)/(len(N)+1e-20)

    if best_val_score < val_score:
        best_epoch = epoch
        best_val_score = val_score
        best_val_auc = val_auc
        # Run on test set
        for idx in N:
            _, auc, acc = test_model(id_=idx, split='test')
            test_scores.append(acc)
            test_aucs.append(auc)
        best_test_score = sum(test_scores)/(len(N)+1e-20)
        best_test_auc = sum(test_aucs)/(len(N)+1e-20)
    #
    print('Test_Epoch: {}; val_scores: {}; val_aucs: {}; test_scores: {}; test_aucs: {}'.format(
        epoch, val_scores, val_aucs, test_scores, test_aucs))
    if params.neptune:
        run["Valid Accuracy"].log(val_score)
        run["Best Test Accuracy"].log(best_test_score)
        run["Best Test Auc"].log(best_test_auc)
        run["Best Valid Accuracy"].log(best_val_score)
        run["Best Valid Auc"].log(best_val_auc)
        run["Best Epoch"].log(best_epoch)
        run["Epoch"].log(epoch)
    # if params.neptune:
    #     neptune.log_metric('Valid Accuracy', val_score)
    #     neptune.log_metric('Best Test Accuracy', best_test_score)
    #     neptune.log_metric('Best Test Auc', best_test_auc)
    #     neptune.log_metric('Best Valid Accuracy', best_val_score)
    #     neptune.log_metric('Best Valid Auc', best_val_auc)
    #     neptune.log_metric('Best Epoch', best_epoch)
    #     neptune.log_metric('Epoch', epoch)


def test_model(id_, split='val'):
    model.eval()
    config['mode'] = 'test'
    if split == 'val':
        valid_dataset.seed = id_
    elif split == 'test':
        test_dataset.seed = id_
    loader = torch.utils.data.DataLoader(
        valid_dataset if split == 'val' else test_dataset, collate_fn=collate_fn, batch_size=params.test_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    total_loss, all_preds, all_targets = 0., [], []
    n_batch = 0
    for batch in loader:
        if sampling == 'unbiased':
            output = run_unbiased(batch, config)
        elif sampling == 'biased':
            output = run_biased(batch, config)
        else:
            output = run_random(batch, config)
        target = batch['output_labels'].float().numpy()
        mask = batch['output_mask'].numpy() == 1
        all_preds.append(output[mask])
        all_targets.append(target[mask])
        n_batch += 1

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return total_loss/n_batch, auc, accuracy


if __name__ == "__main__":
    # Parse command line arguments
    params = create_parser()
    print(params)
    
    # Check if GPU is available when requested
    if params.use_cuda:
        assert device.type == 'cuda', 'no gpu found!'

    # Set up Neptune experiment tracking if enabled
    if params.neptune:
        import neptune
        project = "chloesimanek/BOBCAT"
        run = neptune.init_run(
            project=project,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            name=params.model,
            tags=[params.model, params.dataset],
            source_files=[],
        )
    # if params.neptune:
    #     import neptune
    #     project = "chloesimanek/bobcat"
    #     neptune.init(project_qualified_name=project,
    #                  api_token=os.environ["NEPTUNE_API_TOKEN"])
    #     neptune_exp = neptune.create_experiment(
    #         name=params.file_name, send_hardware_metrics=False, params=vars(params))

    # Initialize configuration and batch_idx for logging
    config = {}
    batch_idx = 0  # #ACTUAL ID - Define batch_idx globally for access in other functions
    
    # Initialize random seeds
    initialize_seeds(params.seed)

    # Parse model type
    base, sampling = params.model.split('-')[0], params.model.split('-')[-1]
    
    # Initialize model
    if base == 'biirt':
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=1).to(device)
        meta_params = [torch.Tensor(
            1, 1).normal_(-1., 1.).to(device).requires_grad_()]
    if base == 'binn':
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=params.question_dim).to(device)
        meta_params = [torch.Tensor(
            1, params.question_dim).normal_(-1., 1.).to(device).requires_grad_()]

    # Initialize optimizers
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-8)
    meta_params_optimizer = torch.optim.SGD(
        meta_params, lr=params.meta_lr, weight_decay=2e-6, momentum=0.9)
    
    # Log model summary
    if params.neptune:
        run["model/summary"] = str(model)
    print(model)

    # Initialize policy components
    if sampling == 'unbiased':
        betas = (0.9, 0.999)
        K_epochs = 4
        eps_clip = 0.2
        memory = Memory()
        ppo_policy = PPO(params.n_question, params.n_question,
                         params.policy_lr, betas, K_epochs, eps_clip)
        if params.neptune:
            run["ppo_model/summary"] = str(ppo_policy.policy)
                
    if sampling == 'biased':
        betas = (0.9, 0.999)
        st_policy = StraightThrough(params.n_question, params.n_question,
                                    params.policy_lr, betas)
        if params.neptune:
            run["biased_model/summary"] = str(st_policy.policy)

    # Load dataset
    data_path = os.path.normpath('data/train_task_'+params.dataset+'.json')
    
    # MODIFIED: Load and store the raw data for reference
    try:
        with open(data_path, 'r') as f:
            raw_train_data = json.load(f)
        print(f"Successfully loaded raw data: {len(raw_train_data)} items")
    except Exception as e:
        print(f"Warning: Could not load raw data directly: {e}")
        raw_train_data = []
    
    train_data, valid_data, test_data = data_split(
        data_path, params.fold, params.seed)
    train_dataset, valid_dataset, test_dataset = Dataset(
        train_data), Dataset(valid_data), Dataset(test_data)
        
    # Setup data loader
    num_workers = 3
    collate_fn = collate_fn(params.n_question)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=params.train_batch_size, 
        num_workers=num_workers, shuffle=True, drop_last=True)
        
    # LOG CODE: Print model and dataset information
    print(f"\nRunning BOBCAT with:")
    print(f"  - Model: {params.model}")
    print(f"  - Dataset: {params.dataset}")
    print(f"  - Questions per student: {params.n_query}")
    print(f"  - Total questions: {params.n_question}")
    print(f"  - Training students: {len(train_data)}")
    
    # Start training timer
    start_time = time.time()
    
    # Training loop
    for epoch in range(params.n_epoch):
        train_model()
        if epoch >= (best_epoch+params.wait):
            break

    # Save final model
    if not os.path.exists('saved_models'): 
        os.makedirs('saved_models') 
        
    model_save_path = f"saved_models/bobcat_final_{params.dataset}_{params.model}_{params.n_query}.pt" 
    torch.save({ 'epoch': epoch, 
        'model_state_dict': model.state_dict(), 
        'meta_params': meta_params, 
        'optimizer_state_dict': optimizer.state_dict(), 
        'meta_optimizer_state_dict': meta_params_optimizer.state_dict(), 
        'val_score': best_val_score, 
        'val_auc': best_val_auc, 
        'test_score': best_test_score, 
        'test_auc': best_test_auc, 
        'params': vars(params) 
    }, model_save_path) 
    print(f"Final model saved to {model_save_path}")
    
    # LOG CODE: Print final results
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_score:.4f} (epoch {best_epoch})")
    print(f"Best test accuracy: {best_test_score:.4f}")
    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")