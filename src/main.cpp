#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <net.h>
#include <chrono>

using namespace cv;
using namespace std;


ncnn::Mat resize(const ncnn::Mat& src, int w, int h)
{
    int src_w = src.w;
    int src_h = src.h;
    unsigned char* u_src = new unsigned char[src_w * src_h * 3];
    src.to_pixels(u_src, ncnn::Mat::PIXEL_RGB);
    unsigned char* u_dst = new unsigned char[w * h * 3];
    ncnn::resize_bilinear_c3(u_src, src_w, src_h, u_dst, w, h);
    ncnn::Mat dst = ncnn::Mat::from_pixels(u_dst, ncnn::Mat::PIXEL_RGB, w, h);
    delete[] u_src;
    delete[] u_dst;
    return dst;
}

vector<string> read_txt(const string& file)
{
    ifstream in;
    in.open(file.data());
    assert(in.is_open());

    vector<string> res;
    string s;
    while(getline(in,s))
    {
        res.push_back(s);
    }
    in.close();
    return res;
}

vector<string> split(const string &s, const char &c)
{
    vector<string> re;
    if (s.empty()) return re;

    string t;
    for (char i : s)
    {
        if (i == c)
        {
            if (!t.empty())
            {
                re.push_back(t);
                t.clear();
            }
            continue;
        }
        t.push_back(i);
    }
    if (!t.empty())
        re.push_back(t);
    return re;
}


int main(int argc, char* argv[])
{
    string model_name = "mobilenetv2";
    if (argc == 2)
    {
        model_name = argv[1];
    }

    string root_path = "../";
    float threshold = 0.6;

    string param_path = root_path + "models/" + model_name + "/ncnn.param";
    string bin_path = root_path + "models/" + model_name + "/ncnn.bin";

    cout << "root dir path = " << root_path << endl;
    cout << "load model name = " << model_name << endl;
    cout << "threshold = " << threshold << endl;

    ncnn::Net net;
    if (0 != net.load_param(param_path.c_str()))
    {
        cout << "load param failed!" << endl;
        return -1;
    }

    if (0 != net.load_model(bin_path.c_str()))
    {
        cout << "load model failed!" << endl;
        return -1;
    }


    string img_lists_path = root_path + "data/imglists/label.txt";
    vector<string> img_lists = read_txt(img_lists_path);
    string img_path = root_path + "data/images/";

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.0078125f, 0.0078125f, 0.0078125f};

    int img_size = 24;
    if (model_name == "rnet")
        img_size = 24;
    else if (model_name == "onet")
        img_size = 48;
    else if (model_name == "mobilenetv2")
        img_size = 224;


    int currect = 0;
    double ave_time = 0.0;
    int sum = img_lists.size();
    for (int i = 0; i < sum; ++i)
    {
        string line = img_lists[i];
        auto tmp = split(line, ' ');
        string name = tmp[0];
        string label = tmp[1];

        Mat image = imread(img_path + name);
        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);
        ncnn::Mat in = resize(ncnn_img, img_size, img_size);
        in.substract_mean_normalize(mean_vals, norm_vals);

        auto start = std::chrono::system_clock::now();
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score;
        ex.extract("cls_prob", score);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        auto t = elapsed_seconds.count() * 1000;
        ave_time += t / sum;

        int cls_label = (score[1] > threshold ? 1 : 0);

        if (to_string(cls_label) == label)
        {
            currect += 1;
        }
        cout << currect * 1.0 / (i + 1) << endl;
    }

    double ratio = currect * 1.0 / sum;
    cout << "currect ratio = " << ratio << endl;
    cout << "average time = " << ave_time << "ms" << endl;

    return 0;
}
