import numpy as np
import re

def apply_map_function(arg_list, rows, cols, func, varName):
    arg_array = np.array(arg_list, dtype=np.float64).reshape(rows, cols)
    
    match = re.search(r'def (\w+)', func)
    if match:
        try:
            exec(func)
            func_name = match.groups()[0]
            func_obj = locals().get(func_name)
            if func_obj:
                res_array = np.vectorize(func_obj)(arg_array)
                return res_array.flatten().tolist()
            else:
                print(f"Function '{func_name}' not found.")
        except Exception as e:
            print(f"Failed to execute function: {str(e)}")
    else:
        print("No function name found")
    
    return []  # Return an empty list if there's an error