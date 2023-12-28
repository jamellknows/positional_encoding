
import numpy as np
import matplotlib.pyplot as plt


def replace_duplicate_with_zeros(list_of_lists):
    seen_lists = set()

    for i, sublist in enumerate(list_of_lists):
        if tuple(sublist) in seen_lists:
            list_of_lists[i] = [0, 0,0]
        else:
            seen_lists.add(tuple(sublist))

    return list_of_lists
 
def getHenonMapEncoding(embedding, x=4):
    x = int(x)
    seq_len = len(embedding)
    x_n = [i for i in range(x)]
    y_n = [i for i in (range(int(x/2)))]  
    list_append = []
    for a in range(0, seq_len):
        list_x = []
        for x in x_n:
            for y in y_n:
                if x == 0:
                    x = 1
                    b = y/x
                b = y/x  
                # print(f" variables {a}, {x}, {y}, {b}")
                x_i = 1 - a * x**2 + y
                list_x.append([x_i,y,b])
                # print(x_i)
                # print([x_i,y])
                # print([x,y])
                
        list_x = replace_duplicate_with_zeros(list_x)
        flattened_list = [item for sublist in list_x for item in sublist]
        list_x = np.array(flattened_list)
        list_append.append(list_x)
        # print(list_x)
        # print(f" list for a x is {list_x}")
        # numpy_x = np.array(list_x)
        # print(f"complete list for numpy {numpy_x}")
    henon_x = np.vstack(list_append)
    return henon_x
        
    
if __name__ == "__main__":
    embedding = [0,1,2,3,4,9,8,7,6,5]
    result = getHenonMapEncoding(embedding)
    print(result[0])
    
# x_n = 1 - a*n^2 + y_n
# y_n = b . x

#  n is chosen 
# x is the dimension d 
# a is the index 
# y is the used mapping ot columns 