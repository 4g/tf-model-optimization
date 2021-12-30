from tensorflow import lite as tflite
import cv2
import numpy as np

class TFLiteModel:
    def load_model(self, model_path):
        self.interpreter = tflite.Interpreter(
            model_path=model_path, num_threads=4)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        return self

    def preprocess(self, image):
        image = cv2.resize(image, (self.width, self.height))
        image = (np.asarray(image, dtype=np.float32))/127.5 - 1.0
        image = np.expand_dims(image, axis=0)
        return image

    def get_model_output(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        outputs = [np.squeeze(x) for x in outputs]
        return outputs

    def get_model_details(self):
        return self.interpreter.get_tensor_details()

    def model_path(self):
        return None

    def speedtest(self, image):
        image = self.preprocess(image)
        from tqdm import tqdm
        for i in tqdm(range(1000)):
            output = self.get_model_output(image)


if __name__ == "__main__":
    import argparse, os, glob, traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="pass either a directory or a tflite model", required=True)

    args = parser.parse_args()

    path = args.model
    try:
        tfutil = TFLiteModel().load_model(path)
        print (f"--------{path}---------")
        print ([(i['name'], i['shape']) for i in tfutil.input_details])
        print ([(i['name'], i['shape']) for i in tfutil.output_details])
        # for x in tfutil.get_model_details():
        #     print (x)

        image = np.zeros((tfutil.width, tfutil.height, 3))
        image = tfutil.preprocess(image)
        from tqdm import tqdm
        for i in tqdm(range(10000)):
            output = tfutil.get_model_output(image)

    except:
        print("Exception in user code:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
