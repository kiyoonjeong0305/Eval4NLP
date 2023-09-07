from model_dict import load_from_catalogue
from guidance_model import Summarization
import argparse
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import os
import json
from utils import score_kendall, load_csv, load_json
import pdb



def make_initial_guide(BPG):
    
    con = BPG.do_guide(aspect="consistency")

    coh = BPG.do_guide(aspect="coherence")

    rel = BPG.do_guide(aspect="relevance")

    return con, coh, rel

def find_penalty(results):
    gt_score = results['score']['gt']
    
    coherence_score = results['score']['coherence']
    consistency_score = results['score']['consistency']
    relevance_score = results['score']['relevance']
    
    coh_con_score = results['score']['coh_con']
    coh_rel_score = results['score']['coh_rel']
    con_rel_score = results['score']['con_rel']
    
    coh_con_rel_score = results['score']['coh_con_rel']
    
    # 가장 높은 스코어를 찾습니다.
    max_score = max(coherence_score, consistency_score, relevance_score, 
                    coh_con_score, coh_rel_score, con_rel_score, 
                    coh_con_rel_score)

    # 가장 높은 스코어에 해당하는 스코어들을 반환합니다.
    if max_score == coherence_score:
        return (0, 1, 1)
    elif max_score == consistency_score:
        return (1, 0, 1)
    elif max_score == relevance_score:
        return (1, 1, 0)
    elif max_score == coh_con_score:
        return (0, 0, 1)
    elif max_score == coh_rel_score:
        return (0, 1, 0)
    elif max_score == con_rel_score:
        return (1, 0, 0)
    elif max_score == coh_con_rel_score:
        return (0, 0, 0)



def penalty_guide(BPG, con, coh, rel, score, penalty_tuple, penalty_prompt):
    if penalty_tuple[0]:
        con = BPG.do_guide(penalty_prompt=penalty_prompt,
                           previous_answer=con,
                           aspect="consistency",
                           )
    else:
        con = con
        
    if penalty_tuple[1]:
        coh = BPG.do_guide(penalty_prompt=penalty_prompt,
                           previous_answer=coh,
                           aspect="coherence",
                           )
    else:
        coh = coh
        
    if penalty_tuple[2]:
        rel = BPG.do_guide(penalty_prompt=penalty_prompt,
                           previous_answer=rel,
                           aspect="relevance",
                           )
    else:
        rel = rel
    return con, coh, rel


def run(consistency,
        coherence, 
        relevance, 
        BPG, 
        data,
        score_options,
        score_prompt,
        guide_prompt,
        penalty_prompt,
        ):
    
    # con, coh, rel = make_initial_guide(BPG, con, coh, rel, score)
    
    final_results = {
        'aspect': {},
        'prompt': {},
        'score': {
            'gt': [],
            'consistency': [],
            'coherence': [],
            'relevance': []
        },
        'kendall_tau': {}
    }
    
    final_results['aspect']['consistency'] = coherence
    final_results['aspect']['coherence'] = consistency
    final_results['aspect']['relevance'] = relevance
    
    final_results['prompt']['guide'] = guide_prompt
    final_results['prompt']['score'] = score_prompt
    final_results['prompt']['penalty'] = penalty_prompt
    
    final_results['score_options'] = score_options
    
    final_results['score']['gt'] = data['Score']
    
    guides = [consistency, coherence, relevance]
    for i in tqdm(range(len(data))):
        try: 
            con_coh_rel = []
            for guide in guides:
                score = BPG.do_score(
                    score_prompt=score_prompt,
                    guide=guide,
                    source=data["SRC"][i],
                    summary=data["HYP"][i],
                )
                con_coh_rel.append(score)
                
            final_results['score']['consistency'].append(con_coh_rel[0])
            final_results['score']['coherence'].append(con_coh_rel[1])
            final_results['score']['relevance'].append(con_coh_rel[2])
            
        except KeyboardInterrupt:
            exit()
        
        except:
            pass
        
    consistency = final_results['score']['consistency']
    coherence = final_results['score']['coherence']
    relevance = final_results['score']['relevance']
    gt_score = final_results['score']['gt']
    
    # pdb.set_trace()
    consistency_score = score_kendall(consistency, gt_score)
    coherence_score = score_kendall(coherence, gt_score)
    relevance_score = score_kendall(relevance, gt_score)
    
    coh_con_score = score_kendall(coherence + consistency)
    coh_rel_score = score_kendall(coherence + relevance)
    con_rel_score = score_kendall(consistency + relevance)
    
    coh_con_rel_score = score_kendall(coherence + consistency + relevance)
    
    final_results['kendall_tau']['coherence'] = coherence_score
    final_results['kendall_tau']['consistency'] = consistency_score
    final_results['kendall_tau']['relevance'] = relevance_score
    final_results['kendall_tau']['coh_con'] = coh_con_score
    final_results['kendall_tau']['coh_rel'] = coh_rel_score
    final_results['kendall_tau']['con_rel'] = con_rel_score
    final_results['kendall_tau']['coh_con_rel'] = coh_con_rel_score
    
    return final_results