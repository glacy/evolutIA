import time
from typing import List, Dict

def original_method(materials: List[Dict]):
    all_exercises = []
    for material in materials:
        for exercise in material['exercises']:
            solution = None
            for sol in material['solutions']:
                if sol['exercise_label'] == exercise['label']:
                    solution = sol
                    break

            all_exercises.append({'exercise': exercise, 'solution': solution})
    return all_exercises

def optimized_method(materials: List[Dict]):
    all_exercises = []
    for material in materials:
        # Preserve first match behavior
        solutions_dict = {}
        for sol in material['solutions']:
            if sol['exercise_label'] not in solutions_dict:
                solutions_dict[sol['exercise_label']] = sol

        for exercise in material['exercises']:
            solution = solutions_dict.get(exercise['label'])
            all_exercises.append({'exercise': exercise, 'solution': solution})
    return all_exercises

# Generate mock data
materials = []
for i in range(10): # 10 materials
    material = {
        'exercises': [{'label': f'ex_{j}'} for j in range(1000)],
        'solutions': [{'exercise_label': f'ex_{j}', 'content': f'sol_{j}'} for j in range(1000)]
    }
    materials.append(material)

start = time.time()
original_method(materials)
end = time.time()
print(f"Original: {end - start:.5f} seconds")

start = time.time()
optimized_method(materials)
end = time.time()
print(f"Optimized: {end - start:.5f} seconds")
