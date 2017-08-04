def run_multiclass(image_path):
    import os, sys

    import tensorflow as tf

    os.environ['TF_CPP_MIN_LqOG_LEVEL'] = '2'

    print (image_path)
    print (type(image_path))
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                       in tf.gfile.GFile("retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor,
                 {'DecodeJpeg/contents:0': image_data})
#Don't need this code
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        #prints result onto console/terminal
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            the_score = '%.5f' % (score)

        # moves the image to its perspective folder
        # !!!!!! Make sure there is a folder for each class!!!!
        classification = " "
        score = predictions[0][top_k[0]]
        new_path = " "
        if score > .7:
            classification = label_lines[top_k[0]]
        else:
            classification = 'unclassified'
        move_to_folder(image_path, classification)

        # writes results into text files
        # It should not be a problem if there is not already a pre-existing text file for each class.
        # It creates on automatically if one does not exist and adds on if one does exist.
        print (classification)
        file = classification + ".txt"
        print (os.path.exists(file))
        f = open(file, "a")
        f.write("\nResults for the photo in this directory: " + new_path + "\n")
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            text = '%s (score = %.5f)  ' % (human_string, score)
            f.write(text)
        f.close()
        print ("wrote scores into " + file)

        '''
        print (os.path.exists(file))
        print ('working')
        f = open('fog.txt', "w")
        f.write("writing?")
        f.close()
'''
    print ("Ran multiclass on the image" + str(image_path))



def rename_if_there_already_exists(new_image_path):
    import os
    print (new_image_path)
    while os.path.exists(new_image_path) == True:
        filename, file_extension = os.path.splitext(new_image_path)
        new_image_path = (filename + "(0)" + file_extension)
        print (new_image_path)
    return new_image_path

def move_to_folder(which_image, categorized_as):
    import os
    image_name = (os.path.basename(which_image))
    directory = "/home/fuego/image_data/" + categorized_as
    new_path = (directory + "/" + image_name)
    # the while loop makes sure there is not already a file that exists with that name and renames it
    rename = rename_if_there_already_exists(new_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.rename(which_image, new_path)
    print ("image moved to " + new_path)