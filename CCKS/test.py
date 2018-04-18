import numpy as np
import tensorflow as tf
a = [[1,2,3],[4,5,6],[7,8,9]]
b = np.array(a)
# print(b[0:2])

# for i in a:
#     print(i)


# other_to_int_table_file = open('./UtilsData/other_to_int_table.txt', 'w', encoding='utf-8')
# test = [6]
# test = test + [7]
# print(test)

test = [[1],[2],[3123]]
a = tf.shape(test)[1]
with tf.Session() as sess:
    print(sess.run(a))
