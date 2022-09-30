import tensorflow, numpy, celluloid, ffmpeg
import os

def check_configuration(print_library_version=True, print_gpu_usage=True):
    print('>'*60)
    print('Check configuration')
    if print_library_version:
        print('Check library versions:')
        print(' numpy == {}'.format(numpy.__version__))
        print(' tensorflow == {}'.format(tensorflow.__version__))
        print(' celluloid == {}'.format(celluloid.__version__))
    if print_gpu_usage:
        if tensorflow.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tensorflow.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    check_configuration()

