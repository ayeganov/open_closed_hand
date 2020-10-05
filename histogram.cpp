//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
//
//using namespace cv;
//
//int main( int argc, char** argv )
//{
//    Mat src, hsv;
//    if( argc != 2 || !(src=imread(argv[1], 1)).data )
//        return -1;
//
//    cvtColor(src, hsv, cv::COLOR_BGR2HSV);
//
//    // Quantize the hue to 30 levels
//    // and the saturation to 32 levels
//    int hbins = 30, sbins = 32;
//    int histSize[] = {hbins, sbins};
//    // hue varies from 0 to 179, see cvtColor
//    float hranges[] = { 0, 180 };
//    // saturation varies from 0 (black-gray-white) to
//    // 255 (pure spectrum color)
//    float sranges[] = { 0, 256 };
//    const float* ranges[] = { hranges, sranges };
//    MatND hist;
//    // we compute the histogram from the 0-th and 1-st channels
//    int channels[] = {0, 1};
//
//    calcHist( &hsv, 1, channels, Mat(), // do not use mask
//             hist, 2, histSize, ranges,
//             true, // the histogram is uniform
//             false );
//    double maxVal=0;
//    minMaxLoc(hist, 0, &maxVal, 0, 0);
//
//    int scale = 10;
//    Mat histImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
//
//    for( int h = 0; h < hbins; h++ )
//        for( int s = 0; s < sbins; s++ )
//        {
//            float binVal = hist.at<float>(h, s);
//            int intensity = cvRound(binVal*255/maxVal);
//            rectangle( histImg, Point(h*scale, s*scale),
//                        Point( (h+1)*scale - 1, (s+1)*scale - 1),
//                        Scalar::all(intensity),
//                        cv::FILLED );
//        }
//
////    namedWindow( "Source", 1 );
////    imshow( "Source", src );
//
//    namedWindow( "H-S Histogram", 1 );
//    imshow( "H-S Histogram", histImg );
//    waitKey();
//}


//#include "opencv2/core/utility.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include <iostream>
//using namespace cv;
//using namespace std;
//int _brightness = 100;
//int _contrast = 100;
//Mat image;
///* brightness/contrast callback function */
//static void updateBrightnessContrast( int /*arg*/, void* )
//{
//    int histSize = 64;
//    int brightness = _brightness - 100;
//    int contrast = _contrast - 100;
//    /*
//     * The algorithm is by Werner D. Streidt
//     * (http://visca.com/ffactory/archives/5-99/msg00021.html)
//     */
//    double a, b;
//    if( contrast > 0 )
//    {
//        double delta = 127.*contrast/100;
//        a = 255./(255. - delta*2);
//        b = a*(brightness - delta);
//    }
//    else
//    {
//        double delta = -128.*contrast/100;
//        a = (256.-delta*2)/255.;
//        b = a*brightness + delta;
//    }
//    Mat dst, hist;
//    image.convertTo(dst, CV_8U, a, b);
//    imshow("image", dst);
//    calcHist(&dst, 1, 0, Mat(), hist, 1, &histSize, 0);
//    Mat histImage = Mat::ones(200, 320, CV_8U)*255;
//    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, CV_32F);
//    histImage = Scalar::all(255);
//    int binW = cvRound((double)histImage.cols/histSize);
//    for( int i = 0; i < histSize; i++ )
//        rectangle( histImage, Point(i*binW, histImage.rows),
//                   Point((i+1)*binW, histImage.rows - cvRound(hist.at<float>(i))),
//                   Scalar::all(0), -1, 8, 0 );
//    imshow("histogram", histImage);
//}
//const char* keys =
//{
//    "{help h||}{@image|baboon.jpg|input image file}"
//};
//int main( int argc, const char** argv )
//{
//    CommandLineParser parser(argc, argv, keys);
//    parser.about("\nThis program demonstrates the use of calcHist() -- histogram creation.\n");
//    if (parser.has("help"))
//    {
//        parser.printMessage();
//        return 0;
//    }
//    string inputImage = parser.get<string>(0);
//    // Load the source image. HighGUI use.
//    image = imread(samples::findFile(inputImage), IMREAD_GRAYSCALE);
//    if(image.empty())
//    {
//        std::cerr << "Cannot read image file: " << inputImage << std::endl;
//        return -1;
//    }
//    namedWindow("image", 0);
//    namedWindow("histogram", 0);
//    createTrackbar("brightness", "image", &_brightness, 200, updateBrightnessContrast);
//    createTrackbar("contrast", "image", &_contrast, 200, updateBrightnessContrast);
//    updateBrightnessContrast(0, 0);
//    waitKey();
//    return 0;
//}


#include "opencv2/core/utility.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctype.h>
using namespace cv;
using namespace std;
Mat image;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;


// User draws box around object to track. This triggers CAMShift to start tracking
static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= Rect(0, 0, image.cols, image.rows);
    }
    switch( event )
    {
    case EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;   // Set up CAMShift properties in main() loop
        break;
    }
}

string hot_keys =
    "\n\nHot keys: \n"
    "\tESC - quit the program\n"
    "\tc - stop the tracking\n"
    "\tb - switch to/from backprojection view\n"
    "\th - show/hide object histogram\n"
    "\tp - pause video\n"
    "To initialize tracking, select the object with mouse\n";
static void help(const char** argv)
{
    cout << "\nThis is a demo that shows mean-shift based tracking\n"
            "You select a color objects such as your face and it tracks it.\n"
            "This reads from video camera (0 by default, or the camera number the user enters\n"
            "Usage: \n\t";
    cout << argv[0] << " [camera number]\n";
    cout << hot_keys;
}
const char* keys =
{
    "{help h | | show help message}{@camera_number| 0 | camera number}"
};
int main( int argc, const char** argv )
{
    VideoCapture cap;
    Rect trackWindow;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    int camNum = parser.get<int>(0);
    cap.open(camNum);
    if( !cap.isOpened() )
    {
        help(argv);
        cout << "***Could not initialize capturing...***\n";
        cout << "Current parameter's value: \n";
        parser.printMessage();
        return -1;
    }
    cout << hot_keys;
    namedWindow( "Histogram", 0 );
    namedWindow( "CamShift Demo", 0 );
    setMouseCallback( "CamShift Demo", onMouse, 0 );
    createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 );
    createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
    createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );

    // histimg = np.zeros((200, 320, 3), dtype=np.dtype("uint8"))
    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    bool paused = false;
    for(;;)
    {
        if( !paused )
        {
            cap >> frame;
            if( frame.empty() )
                break;
        }
        frame.copyTo(image);
        if( !paused )
        {
            cvtColor(image, hsv, COLOR_BGR2HSV);
            if( trackObject )
            {
                int _vmin = vmin, _vmax = vmax;

                // cv2.inRange
                inRange(hsv,
                        Scalar(0, smin, MIN(_vmin,_vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)),
                        mask);

                int ch[] = {0, 0};
                // hue = np.zeros(hsv.shape)
                hue.create(hsv.size(), hsv.depth());
                // cv2.mixChannels
                mixChannels(&hsv, 1, &hue, 1, ch, 1);
                if( trackObject < 0 )
                {
                    // Object has been selected by user, set up CAMShift search properties once
                    // hue[yleft:yright, xleft:xright]
                    // mask[yleft:yright, xleft:xright]
                    Mat roi(hue, selection), maskroi(mask, selection);
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, NORM_MINMAX);
                    trackWindow = selection;
                    trackObject = 1; // Don't set up again, unless user selects new ROI
                    // histimg = np.zeros(histimg.shape)
                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);

                    // depth_uint8 = depth_array*depth_scale_factor+depth_scale_beta_factor
                    // depth_uint8[depth_uint8>255] = 255
                    // depth_uint8[depth_uint8<0] = 0
                    // depth_uint8 = depth_uint8.astype('uint8')

                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    cvtColor(buf, buf, COLOR_HSV2BGR);
                    for( int i = 0; i < hsize; i++ )
                    {
                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                        rectangle( histimg, Point(i*binW,histimg.rows),
                                   Point((i+1)*binW,histimg.rows - val),
                                   Scalar(buf.at<Vec3b>(i)), -1, 8 );
                    }
                }
                // Perform CAMShift
                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);

                // np.logical_and(backproj, mask)
                backproj &= mask;
                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                    TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
                if( trackWindow.area() <= 1 )
                {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }
                if( backprojMode )
                    cvtColor( backproj, image, COLOR_GRAY2BGR );
                ellipse( image, trackBox, Scalar(0,0,255), 3, LINE_AA );
            }
        }
        else if( trackObject < 0 )
            paused = false;
        if( selectObject && selection.width > 0 && selection.height > 0 )
        {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }
        imshow( "CamShift Demo", image );
        imshow( "Histogram", histimg );
        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch(c)
        {
        case 'b':
            backprojMode = !backprojMode;
            break;
        case 'c':
            trackObject = 0;
            histimg = Scalar::all(0);
            break;
        case 'h':
            showHist = !showHist;
            if( !showHist )
                destroyWindow( "Histogram" );
            else
                namedWindow( "Histogram", 1 );
            break;
        case 'p':
            paused = !paused;
            break;
        default:
            ;
        }
    }
    return 0;
}

