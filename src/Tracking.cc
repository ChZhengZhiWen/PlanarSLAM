#include "Tracking.h"
#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include <iostream>
#include <mutex>
#include "PlaneExtractor.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM {
    PlaneDetection plane_detection;

    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
                       KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor) :
            mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
            mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys),
            mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0) {
        // Load camera parameters from settings file

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        //
        int img_width = fSettings["Camera.width"];
        int img_height = fSettings["Camera.height"];

        cout << "img_width = " << img_width << endl<< "img_height = " << img_height << endl;

        initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3, 3), mK, Size(img_width, img_height), CV_32F,
                                mUndistX, mUndistY);

        cout << "mUndistX size = " << mUndistX.size << "mUndistY size = " << mUndistY.size << endl;

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        if (sensor == System::STEREO || sensor == System::RGBD) {
            mThDepth = mbf * (float) fSettings["ThDepth"] / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }

        if (sensor == System::RGBD) {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }

        mfDThRef = fSettings["Plane.AssociationDisRef"];
        mfDThMon = fSettings["Plane.AssociationDisMon"];
        mfAThRef = fSettings["Plane.AssociationAngRef"];
        mfAThMon = fSettings["Plane.AssociationAngMon"];

        mfVerTh = fSettings["Plane.VerticalThreshold"];
        mfParTh = fSettings["Plane.ParallelThreshold"];

        manhattanCount = 0;
        fullManhattanCount = 0;

        mpPointCloudMapping = make_shared<MeshViewer>(mpMap);
    }


    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer) {
        mpViewer = pViewer;
    }

    cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp) {
        mImRGB = imRGB;
        mImGray = imRGB;
        mImDepth = imD;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }


        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        mCurrentFrame = Frame(mImRGB, mImGray, mImDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK,
                              mDistCoef, mbf, mThDepth, mDepthMapFactor);
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();


        ////Track();
        Track_zzw();

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        double featureT = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t1).count();
        double trackT = getTrackTime();

        std::ofstream fileWrite12("total_plp.txt", std::ios::app);
        std::ofstream fileWriteTrack("Track_plp.txt", std::ios::app);
        std::ofstream fileWriteFeature("Feature_plp.txt", std::ios::app);
        fileWrite12<<t12<<endl;
        fileWriteTrack<<trackT<<endl;
        fileWriteFeature<<featureT<<endl;
        //fileWrite12.write((char*) &trackT, sizeof(double));
        fileWrite12.close();
        fileWriteTrack.close();
        fileWriteFeature.close();

//        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
//        double t32= std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
        return mCurrentFrame.mTcw.clone();
    }

    void Tracking::Track_zzw() {

        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }

        double ttrack_12 = 0.0;
        double ttrack_34 = 0.0;
        trackTime = 0.0;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        //当有效时,lock锁管理模板类管理锁对象，周期结束后自动解锁 ，意思为锁住地图更新
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        if (mState == NOT_INITIALIZED) {
            if (mSensor == System::STEREO || mSensor == System::RGBD) {
                Rotation_cm = cv::Mat::zeros(cv::Size(3, 3), CV_32F);

                StereoInitialization();
                Rotation_cm = mpMap->FindManhattan(mCurrentFrame, mfVerTh, true);
                //Rotation_cm=SeekManhattanFrame(mCurrentFrame.vSurfaceNormal,mCurrentFrame.mVF3DLines).clone();
                Rotation_cm = TrackManhattanFrame(Rotation_cm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();
                mLastRcm = Rotation_cm.clone();

            } else
                MonocularInitialization();
            //更新帧绘制器中存储的最新状态
            mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        } else {
            //Tracking: system is initialized
            bool bOK = false;
            bool bManhattan = false;
            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            // mbOnlyTracking等于false表示正常SLAM模式（定位+地图更新），mbOnlyTracking等于true表示仅定位模式
            // tracking 类构造时默认为false。有个开关ActivateLocalizationMode，可以控制是否开启mbOnlyTracking
            if (!mbOnlyTracking) {
                bManhattan = DetectManhattan();
                if (bManhattan){
                    char one;
                    one = getchar();
                }

                mUpdateMF = true;
                cv::Mat MF_can = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                cv::Mat MF_can_T = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                MF_can = TrackManhattanFrame(mLastRcm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();

//                cout<<endl<<"MF_can00000000000000000000000000000"<<MF_can<<endl;

                MF_can.copyTo(mLastRcm);//.clone();
                MF_can_T = MF_can.t();
                mRotation_wc = Rotation_cm * MF_can_T;
                mRotation_wc = mRotation_wc.t();



                //检查并更新上一帧被替换的MapPoints
                //局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
                CheckReplacedInLastFrame();

                // 运动模型是空的或刚完成重定位，跟踪参考关键帧；否则恒速模型跟踪
                // 第一个条件,如果运动模型为空,说明是刚初始化开始，或者已经跟丢了
                // 第二个条件,如果当前帧紧紧地跟着在重定位的帧的后面，我们将重定位帧来恢复位姿
                // mnLastRelocFrameId 上一次重定位的那一帧
                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                    bOK = TranslationEstimation();

                } else {
                    // 用最近的普通帧来跟踪当前的普通帧
                    // 根据恒速模型设定当前帧的初始位姿
                    // 通过投影的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点所对应3D点的投影误差即可得到位姿
                    bOK = TranslationWithMotionModel();
                    if (!bOK) {
                        //根据恒速模型失败了，只能根据TranslationEstimation来跟踪
                        bOK = TranslationEstimation();
                    }
                }
            }

            // 将最新的关键帧作为当前帧的参考关键帧
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            // 在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
            // 前面只是跟踪一帧得到初始位姿，这里搜索局部关键帧、局部地图点，和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
            if (!mbOnlyTracking) {
                if (bOK) {
                    bOK = TrackLocalMap();
                } else {
                    bOK = Relocalization();
                }
            }

            // update rotation from manhattan
            cv::Mat new_Rotation_wc = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat Rotation_mc = Rotation_cm.t();
            cv::Mat MF_can_T;
            MF_can_T = Rotation_mc * new_Rotation_wc;
            mLastRcm = MF_can_T.t();

            if (bOK)
                mState = OK;
            else
                mState = LOST;

            // Update drawer
            mpFrameDrawer->Update(this);

            mpMap->FlagMatchedPlanePoints(mCurrentFrame, mfDThRef);

            //Update Planes
            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
                if (pMP) {
                    pMP->UpdateCoefficientsAndPoints(mCurrentFrame, i);
                } else if (!mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mbNewPlane = true;
                }
            }

            mpPointCloudMapping->print();

            // If tracking were good, check if we insert a keyframe
            if (bOK) {
                // Update motion model
                //跟踪成功，更新恒速运动模型
                if (!mLastFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    // mVelocity = Tcl = Tcw * Twl,表示上一帧到当前帧的变换， 其中 Twl = LastTwc
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                } else
                    mVelocity = cv::Mat();

                //更新显示中的位姿
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
                // Clean VO matches
                //清除观测不到的地图点
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }
                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    if (pML)
                        if (pML->Observations() < 1) {
                            mCurrentFrame.mvbLineOutlier[i] = false;
                            mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                        }
                }
                for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
                    MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbPlaneOutlier[i] = false;
                            mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(NULL);
                        }

                    MapPlane *pVMP = mCurrentFrame.mvpVerticalPlanes[i];
                    if (pVMP)
                        if (pVMP->Observations() < 1) {
                            mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                            mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(NULL);
                        }

                    MapPlane *pPMP = mCurrentFrame.mvpParallelPlanes[i];
                    if (pVMP)
                        if (pVMP->Observations() < 1) {
                            mCurrentFrame.mvbParPlaneOutlier[i] = false;
                            mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                // 清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）
                // Clean VO matches 只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
                // 临时地图点仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                     lit != lend; lit++) {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                for (list<MapLine *>::iterator lit = mlpTemporalLines.begin(), lend = mlpTemporalLines.end();
                     lit != lend; lit++) {
                    MapLine *pML = *lit;
                    delete pML;
                }
                // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
                // 不能够直接执行这个是因为其中存储的都是指针,之前的操作都是为了避免内存泄露
                mlpTemporalPoints.clear();
                mlpTemporalLines.clear();

                std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();


                // Check if we need to insert a new keyframe
                // 检测并插入关键帧，对于双目或RGB-D会产生新的地图点
                if (NeedNewKeyFrame()) {
                    CreateNewKeyFrame();
                }

                std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();

                ttrack_34= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();


                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                // 作者这里说允许在BA中被Huber核函数判断为外点的传入新的关键帧中，让后续的BA来审判他们是不是真正的外点
                // 但是估计下一帧位姿的时候我们不想用这些外点，所以删掉
                // 删除那些在bundle adjustment中检测为outlier的地图点
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    // 这里第一个条件还要执行判断是因为, 前面的操作中可能删除了其中的地图点
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    if (mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            // 如果初始化后不久就跟踪失败，并且relocation也没有搞定，只能重新Reset
            if (mState == LOST) {
                if (mpMap->KeyFramesInMap() <= 5) {
                    mpSystem->Reset();
                    return;
                }
            }

            //确保已经设置了参考关键帧
            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // 保存上一帧的数据,当前帧变上一帧
            mLastFrame = Frame(mCurrentFrame);
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        ttrack_12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        trackTime = ttrack_12 - ttrack_34;

        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        // 记录位姿信息，用于最后保存所有的轨迹
        if (!mCurrentFrame.mTcw.empty()) {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        } else {
            // This can happen if tracking is lost
            // 如果跟踪失败，则相对位姿使用上一次值
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }

    }

    void Tracking::Track() {

        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }

        double ttrack_12 = 0.0;
        double ttrack_34 = 0.0;
        trackTime = 0.0;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        //当有效时,lock锁管理模板类管理锁对象，周期结束后自动解锁 ，意思为锁住地图更新
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        if (mState == NOT_INITIALIZED) {
            if (mSensor == System::STEREO || mSensor == System::RGBD) {
                Rotation_cm = cv::Mat::zeros(cv::Size(3, 3), CV_32F);

                StereoInitialization();
                Rotation_cm = mpMap->FindManhattan(mCurrentFrame, mfVerTh, true);
                //Rotation_cm=SeekManhattanFrame(mCurrentFrame.vSurfaceNormal,mCurrentFrame.mVF3DLines).clone();
                Rotation_cm = TrackManhattanFrame(Rotation_cm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();
                mLastRcm = Rotation_cm.clone();

            } else
                MonocularInitialization();
            //更新帧绘制器中存储的最新状态
            mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        } else {
            //Tracking: system is initialized
            bool bOK = false;
            bool bManhattan = false;
            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            // mbOnlyTracking等于false表示正常SLAM模式（定位+地图更新），mbOnlyTracking等于true表示仅定位模式
            // tracking 类构造时默认为false。有个开关ActivateLocalizationMode，可以控制是否开启mbOnlyTracking
            if (!mbOnlyTracking) {
                bManhattan = DetectManhattan();
                if (bManhattan){
                    char one;
                    one = getchar();
                }

                mUpdateMF = true;
                cv::Mat MF_can = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                cv::Mat MF_can_T = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                MF_can = TrackManhattanFrame(mLastRcm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();

//                cout<<endl<<"MF_can00000000000000000000000000000"<<MF_can<<endl;

                MF_can.copyTo(mLastRcm);//.clone();
                MF_can_T = MF_can.t();
                mRotation_wc = Rotation_cm * MF_can_T;
                mRotation_wc = mRotation_wc.t();



                //检查并更新上一帧被替换的MapPoints
                //局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
                CheckReplacedInLastFrame();

                // 运动模型是空的或刚完成重定位，跟踪参考关键帧；否则恒速模型跟踪
                // 第一个条件,如果运动模型为空,说明是刚初始化开始，或者已经跟丢了
                // 第二个条件,如果当前帧紧紧地跟着在重定位的帧的后面，我们将重定位帧来恢复位姿
                // mnLastRelocFrameId 上一次重定位的那一帧
                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                    bOK = TranslationEstimation();

                } else {
                    // 用最近的普通帧来跟踪当前的普通帧
                    // 根据恒速模型设定当前帧的初始位姿
                    // 通过投影的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点所对应3D点的投影误差即可得到位姿
                    bOK = TranslationWithMotionModel();
                    if (!bOK) {
                        //根据恒速模型失败了，只能根据TranslationEstimation来跟踪
                        bOK = TranslationEstimation();
                    }
                }
            }

            // 将最新的关键帧作为当前帧的参考关键帧
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            // 在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
            // 前面只是跟踪一帧得到初始位姿，这里搜索局部关键帧、局部地图点，和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
            if (!mbOnlyTracking) {
                if (bOK) {
                    bOK = TrackLocalMap();
                } else {
                    bOK = Relocalization();
                }
            }

            // update rotation from manhattan
            cv::Mat new_Rotation_wc = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat Rotation_mc = Rotation_cm.t();
            cv::Mat MF_can_T;
            MF_can_T = Rotation_mc * new_Rotation_wc;
            mLastRcm = MF_can_T.t();

            if (bOK)
                mState = OK;
            else
                mState = LOST;

            // Update drawer
            mpFrameDrawer->Update(this);

            mpMap->FlagMatchedPlanePoints(mCurrentFrame, mfDThRef);

            //Update Planes
            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
                if (pMP) {
                    pMP->UpdateCoefficientsAndPoints(mCurrentFrame, i);
                } else if (!mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mbNewPlane = true;
                }
            }

            mpPointCloudMapping->print();

            // If tracking were good, check if we insert a keyframe
            if (bOK) {
                // Update motion model
                //跟踪成功，更新恒速运动模型
                if (!mLastFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    // mVelocity = Tcl = Tcw * Twl,表示上一帧到当前帧的变换， 其中 Twl = LastTwc
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                } else
                    mVelocity = cv::Mat();

                //更新显示中的位姿
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
                // Clean VO matches
                //清除观测不到的地图点
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }
                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    if (pML)
                        if (pML->Observations() < 1) {
                            mCurrentFrame.mvbLineOutlier[i] = false;
                            mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                        }
                }
                for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
                    MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbPlaneOutlier[i] = false;
                            mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(NULL);
                        }

                    MapPlane *pVMP = mCurrentFrame.mvpVerticalPlanes[i];
                    if (pVMP)
                        if (pVMP->Observations() < 1) {
                            mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                            mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(NULL);
                        }

                    MapPlane *pPMP = mCurrentFrame.mvpParallelPlanes[i];
                    if (pVMP)
                        if (pVMP->Observations() < 1) {
                            mCurrentFrame.mvbParPlaneOutlier[i] = false;
                            mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                // 清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）
                // Clean VO matches 只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
                // 临时地图点仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                     lit != lend; lit++) {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                for (list<MapLine *>::iterator lit = mlpTemporalLines.begin(), lend = mlpTemporalLines.end();
                     lit != lend; lit++) {
                    MapLine *pML = *lit;
                    delete pML;
                }
                // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
                // 不能够直接执行这个是因为其中存储的都是指针,之前的操作都是为了避免内存泄露
                mlpTemporalPoints.clear();
                mlpTemporalLines.clear();

                std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();


                // Check if we need to insert a new keyframe
                // 检测并插入关键帧，对于双目或RGB-D会产生新的地图点
                if (NeedNewKeyFrame()) {
                    CreateNewKeyFrame();
                }

                std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();

                ttrack_34= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();


                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                // 作者这里说允许在BA中被Huber核函数判断为外点的传入新的关键帧中，让后续的BA来审判他们是不是真正的外点
                // 但是估计下一帧位姿的时候我们不想用这些外点，所以删掉
                // 删除那些在bundle adjustment中检测为outlier的地图点
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    // 这里第一个条件还要执行判断是因为, 前面的操作中可能删除了其中的地图点
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    if (mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            // 如果初始化后不久就跟踪失败，并且relocation也没有搞定，只能重新Reset
            if (mState == LOST) {
                if (mpMap->KeyFramesInMap() <= 5) {
                    mpSystem->Reset();
                    return;
                }
            }

            //确保已经设置了参考关键帧
            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // 保存上一帧的数据,当前帧变上一帧
            mLastFrame = Frame(mCurrentFrame);
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        ttrack_12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        trackTime = ttrack_12 - ttrack_34;

        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        // 记录位姿信息，用于最后保存所有的轨迹
        if (!mCurrentFrame.mTcw.empty()) {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        } else {
            // This can happen if tracking is lost
            // 如果跟踪失败，则相对位姿使用上一次值
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }

    }


    double Tracking::getTrackTime()
    {
        return trackTime;
    }

    cv::Mat
    Tracking::SeekManhattanFrame(vector<SurfaceNormal> &vTempSurfaceNormal, vector<FrameLine> &vVanishingDirection) {


        vector<cv::Mat> vRotaionMatrix;
        vector<cv::Mat> vRotaionMatrix_good;
        cv::RNG rnger(cv::getTickCount());
        vector<cv::Mat> vSN_good;
        vector<double> lambda_good;
        vector<cv::Point2d> m_j_selected;
        // R_cm_update matrix
        cv::Mat R_cm_update=cv::Mat::eye(cv::Size(3,3),CV_32F);
        cv::Mat R_cm_new=cv::Mat::eye(cv::Size(3,3),CV_32F);
// initialization with random matrix
#if 1
        cv::Mat qu = cv::Mat::zeros(cv::Size(4,1),CV_32F);
        rnger.fill(qu, cv::RNG::UNIFORM, cv::Scalar::all(0.01), cv::Scalar::all(1));
        Eigen::Quaterniond qnorm;
        Eigen::Quaterniond q(qu.at<float>(0,0),qu.at<float>(1,0),qu.at<float>(2,0),qu.at<float>(3,0));//=Eigen::MatrixXd::Random(1, 4);
        qnorm.x()=q.x()/q.norm();qnorm.y()=q.y()/q.norm();
        qnorm.z()=q.z()/q.norm();qnorm.w()=q.w()/q.norm();
        cv::eigen2cv(qnorm.matrix(),R_cm_update);//	eigen2cv(m, img);;*/
        //cout<<R_cm_update<<endl;
        cv::SVD svd; cv::Mat U,W,VT;
        svd.compute(R_cm_update,W,U,VT);
        R_cm_update=U*VT;
        //cout<<000<<R_cm_update<<endl;
        // R_cm_Rec matrix
        cv::Mat R_cm_Rec=cv::Mat::zeros(cv::Size(3,3),CV_32F);

        cv::Mat R_cm_initial;
        int  validMF=0;
        //cout<<R_cm_update<<endl;
        R_cm_new.at<float>(0,0) = R_cm_update.at<double>(0,0);
        R_cm_new.at<float>(0,1) = R_cm_update.at<double>(0,1);
        R_cm_new.at<float>(0,2) = R_cm_update.at<double>(0,2);
        R_cm_new.at<float>(1,0) = R_cm_update.at<double>(1,0);
        R_cm_new.at<float>(1,1) = R_cm_update.at<double>(1,1);
        R_cm_new.at<float>(1,2) = R_cm_update.at<double>(1,2);
        R_cm_new.at<float>(2,0) = R_cm_update.at<double>(2,0);
        R_cm_new.at<float>(2,1) = R_cm_update.at<double>(2,1);
        R_cm_new.at<float>(2,2) = R_cm_update.at<double>(2,2);
        //cout<<R_cm_new<<endl;
        //matTemp.convertTo(MatTemp2, CV_8U)
#endif

        R_cm_new = TrackManhattanFrame(R_cm_new, vTempSurfaceNormal,vVanishingDirection);
        return R_cm_new;//vRotaionMatrix_good[0];
    }


    cv::Mat Tracking::ClusterMultiManhattanFrame(vector<cv::Mat> &vRotationCandidate, double &clusterRatio) {
        //MF_nonRd = [];
        vector<vector<int>> bin;
        //succ_rate = [];
        cv::Mat a;
        vector<cv::Mat> MF_nonRd;
        int histStart = 0;
        float histStep = 0.1;
        int histEnd = 2;
        int HasPeak = 1;
        int numMF_can = vRotationCandidate.size();
        int numMF = numMF_can;
        //rng(0,'twister');
        int numMF_nonRd = 0;

        while (HasPeak == 1) {
            //随机的一个Rotation
            cv::Mat R = vRotationCandidate[rand() % (numMF_can - 1) + 1];
            cv::Mat tempAA;
            vector<cv::Point3f> Paa;
            //
            vector<float> vDistanceOfRotation;
            cv::Mat Rvec = R.t() * R;
            float theta = acos((trace(Rvec)[0] - 1) / 2);
            cv::Point3f w;
            w.x = theta * 1 / 2 * sin(theta) * (Rvec.at<float>(2, 1) - Rvec.at<float>(1, 2));
            w.y = theta * 1 / 2 * sin(theta) * (Rvec.at<float>(0, 2) - Rvec.at<float>(2, 0));
            w.z = theta * 1 / 2 * sin(theta) * (Rvec.at<float>(1, 0) - Rvec.at<float>(0, 1));

            for (int i = 0; i < vRotationCandidate.size(); i++) {
                cv::Mat RvecBetween = R.t() * vRotationCandidate[i];
                float theta = acos((trace(RvecBetween)[0] - 1) / 2);
                cv::Point3f wb;
                wb.x = theta * 1 / 2 * sin(theta) * (RvecBetween.at<float>(2, 1) - RvecBetween.at<float>(1, 2));
                wb.y = theta * 1 / 2 * sin(theta) * (RvecBetween.at<float>(0, 2) - RvecBetween.at<float>(2, 0));
                wb.z = theta * 1 / 2 * sin(theta) * (RvecBetween.at<float>(1, 0) - RvecBetween.at<float>(0, 1));
                Paa.push_back(wb);
                vDistanceOfRotation.push_back(norm(w - wb));
            }

            //
            bin = EasyHist(vDistanceOfRotation, histStart, histStep, histEnd);

            HasPeak = 0;
            for (int k = 0; k < bin.size(); k++) {
                int binSize = 0;
                for (int n = 0; n < bin[k].size(); n++) { if (bin[k][n] > 0)binSize++; }
                if (binSize / numMF_can > clusterRatio) {
                    HasPeak = 1;
                    break;
                }
            }
            //if(HasPeak == 0) return;

            int binSize1 = 1;
            for (int n = 0; n < bin[0].size(); n++) { if (bin[0][n] > 0)binSize1++; }
            // check whether the dominant bin happens at zero
            if (binSize1 / numMF_can > clusterRatio) {
                cv::Point3f meanPaaTem(0, 0, 0);
                for (int n = 0; n < bin[0].size(); n++) {
                    if (bin[0][n] > 0) {
                        meanPaaTem += Paa[bin[0][n]];
                        meanPaaTem = meanPaaTem / binSize1;
                        meanPaaTem = meanPaaTem / norm(meanPaaTem);
                    }

                }
                //calculate the mean
                float s = sin(norm(meanPaaTem));
                float c = cos(norm(meanPaaTem));
                float t = 1 - c;
                cv::Point3f vec_n(0, 0, 0);
                if (norm(meanPaaTem) <= 0.0001) {}

                else
                    vec_n = meanPaaTem;


                float x = vec_n.x;
                float y = vec_n.y;
                float z = vec_n.z;
                cv::Mat mm = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                mm.at<float>(0, 0) = t * x * x + c;
                mm.at<float>(0, 1) = t * x * y - s * z;
                mm.at<float>(0, 2) = t * x * z + s * y;
                mm.at<float>(1, 0) = t * x * y + s * z;
                mm.at<float>(1, 1) = t * y * y + c;
                mm.at<float>(1, 2) = t * y * z - s * x;
                mm.at<float>(2, 0) = t * x * z - s * y;
                mm.at<float>(2, 1) = t * y * z + s * x;
                mm.at<float>(2, 2) = t * z * z + c;

                if (isnan(sum(mm)[0]) && (norm(meanPaaTem) == 0)) {
                    numMF_nonRd += 1;
                    MF_nonRd.push_back(R);
                } else {
                    numMF_nonRd = numMF_nonRd + 1;
                    MF_nonRd.push_back(R * mm);
                    //succ_rate{numMF_nonRd} = numel(bin{1})/numMF_can;
                }

                /*for(int j = 0;j<bin[0].size();j++)
            {
                if(bin[0][j]>0)
                {
                    vRotationCandidate[];
                }

            }*/
                //MF_can{bin{1}(j)} = [];

            }

        }
        return a;
    }

    vector<vector<int>> Tracking::EasyHist(vector<float> &vDistance, int &histStart, float &histStep, int &histEnd) {
        int numData = vDistance.size();
        int numBin = (histEnd - histEnd) / histStep;
        vector<vector<int>> bin(numBin, vector<int>(numBin, 0));//bin(numBin,0);
        for (int i = 0; i < numBin; i++) {
            float down = (i - 1) * histStep + histStart;
            float up = down + histStep;
            for (int j = 1; j < numData; j++) {
                if (vDistance[j] >= down && vDistance[j] < up)
                    bin[i].push_back(j);//=bin[i]+1;
            }

        }
        return bin;

    }

    cv::Mat Tracking::ProjectSN2MF(int a, const cv::Mat &R_cm, const vector<SurfaceNormal> &vTempSurfaceNormal,
                                   vector<FrameLine> &vVanishingDirection) {
        vector<cv::Point2d> m_j_selected;
        cv::Mat R_cm_Rec = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        cv::Mat R_cm_NULL = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        cv::Mat R_mc = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
        //R_Mc = [R_cM(:,mod(a+3,3)+1), R_cM(:,mod(a+4,3)+1), R_cM(:,mod(a+5,3)+1)].';

        int c1 = (a + 3) % 3;
        int c2 = (a + 4) % 3;
        int c3 = (a + 5) % 3;
        R_mc.at<float>(0, 0) = R_cm.at<float>(0, c1);
        R_mc.at<float>(0, 1) = R_cm.at<float>(0, c2);
        R_mc.at<float>(0, 2) = R_cm.at<float>(0, c3);
        R_mc.at<float>(1, 0) = R_cm.at<float>(1, c1);
        R_mc.at<float>(1, 1) = R_cm.at<float>(1, c2);
        R_mc.at<float>(1, 2) = R_cm.at<float>(1, c3);
        R_mc.at<float>(2, 0) = R_cm.at<float>(2, c1);
        R_mc.at<float>(2, 1) = R_cm.at<float>(2, c2);
        R_mc.at<float>(2, 2) = R_cm.at<float>(2, c3);
        R_mc = R_mc.t();




        //cout<<"R_cm"<<R_cm<<endl;

        /*cout<<"RCM"<<R_cm.at<float>(0,c1)<<", "<<R_cm.at<float>(1,c1)<<","<<R_cm.at<float>(2,c1)<<","<<
            R_cm.at<float>(0,c2)<<","<<R_cm.at<float>(1,c2)<<","<<R_cm.at<float>(2,c2)<<","<<
            R_cm.at<float>(0,c3)<<","<<R_cm.at<float>(1,c3)<<","<<R_cm.at<float>(2,c3)<<endl;*/
        size_t sizeOfSurfaceNormal = vTempSurfaceNormal.size() + vVanishingDirection.size();
        //cout<<"size of SN"<<sizeOfSurfaceNormal<<endl;
        for (size_t i = 0; i < sizeOfSurfaceNormal; i++) {
            cv::Mat temp = cv::Mat::zeros(cv::Size(1, 3), CV_32F);

            if (i >= vTempSurfaceNormal.size()) {
                temp.at<float>(0, 0) = vVanishingDirection[i].direction.x;
                temp.at<float>(1, 0) = vVanishingDirection[i].direction.y;
                temp.at<float>(2, 0) = vVanishingDirection[i].direction.z;
            } else {
                temp.at<float>(0, 0) = vTempSurfaceNormal[i].normal.x;
                temp.at<float>(1, 0) = vTempSurfaceNormal[i].normal.y;
                temp.at<float>(2, 0) = vTempSurfaceNormal[i].normal.z;
            }
            //cout<<temp<<endl;
            //cout<<" TEMP"<<vTempSurfaceNormal[i].x<<","<<vTempSurfaceNormal[i].y<<","<<vTempSurfaceNormal[i].z<<endl;

            cv::Point3f n_ini;
            cv::Mat m_ini;
            m_ini = R_mc * temp;
            n_ini.x = m_ini.at<float>(0, 0);
            n_ini.y = m_ini.at<float>(1, 0);
            n_ini.z = m_ini.at<float>(2, 0);
            /*n_ini.x=R_mc.at<float>(0,0)*temp.at<float>(0,0)+R_mc.at<float>(0,1)*temp.at<float>(1,0)+R_mc.at<float>(0,2)*temp.at<float>(2,0);
        n_ini.y=R_mc.at<float>(1,0)*temp.at<float>(0,0)+R_mc.at<float>(1,1)*temp.at<float>(1,0)+R_mc.at<float>(1,2)*temp.at<float>(2,0);
        n_ini.z=R_mc.at<float>(2,0)*temp.at<float>(0,0)+R_mc.at<float>(2,1)*temp.at<float>(1,0)+R_mc.at<float>(2,2)*temp.at<float>(2,0);
        //cout<<"R_mc"<<R_mc<<endl;*/


            double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
            //cout<<lambda<<endl;
            if (lambda < sin(0.2618)) //0.25
            {
                double tan_alfa = lambda / std::abs(n_ini.z);
                double alfa = asin(lambda);
                double m_j_x = alfa / tan_alfa * n_ini.x / n_ini.z;
                double m_j_y = alfa / tan_alfa * n_ini.y / n_ini.z;
                if (!std::isnan(m_j_x) && !std::isnan(m_j_y))
                    m_j_selected.push_back(cv::Point2d(m_j_x, m_j_y));
                if (i < vTempSurfaceNormal.size()) {
                    if (a == 1)mCurrentFrame.vSurfaceNormalx.push_back(vTempSurfaceNormal[i].FramePosition);
                    else if (a == 2)mCurrentFrame.vSurfaceNormaly.push_back(vTempSurfaceNormal[i].FramePosition);
                    else if (a == 3)mCurrentFrame.vSurfaceNormalz.push_back(vTempSurfaceNormal[i].FramePosition);
                } else {
                    if (a == 1) {
                        cv::Point2d endPoint = vVanishingDirection[i].p;
                        cv::Point2d startPoint = vVanishingDirection[i].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinex.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[i].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCx.push_back(vVanishingDirection[i].rndpts3d[k]);
                    } else if (a == 2) {
                        cv::Point2d endPoint = vVanishingDirection[i].p;
                        cv::Point2d startPoint = vVanishingDirection[i].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLiney.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[i].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCy.push_back(vVanishingDirection[i].rndpts3d[k]);
                    } else if (a == 3) {
                        cv::Point2d endPoint = vVanishingDirection[i].p;
                        cv::Point2d startPoint = vVanishingDirection[i].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinez.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[i].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCz.push_back(vVanishingDirection[i].rndpts3d[k]);
                    }

                }
                //lambda_good.push_back(lambda);
                //找到一个面
            }
        }
        //cout<<"m_j_selected.push_back(temp)"<<m_j_selected.size()<<endl;

        if (m_j_selected.size() > sizeOfSurfaceNormal / 20) {
            //cv::Point2d s_j = MeanShift(m_j_selected);
            sMS tempMeanShift = MeanShift(m_j_selected);
            cv::Point2d s_j = tempMeanShift.centerOfShift;// MeanShift(m_j_selected);
            float s_j_density = tempMeanShift.density;
            //cout<<"tracking:s_j"<<s_j.x<<","<<s_j.y<<endl;
            float alfa = norm(s_j);
            float ma_x = tan(alfa) / alfa * s_j.x;
            float ma_y = tan(alfa) / alfa * s_j.y;
            cv::Mat temp1 = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
            temp1.at<float>(0, 0) = ma_x;
            temp1.at<float>(1, 0) = ma_y;
            temp1.at<float>(2, 0) = 1;

            R_cm_Rec = R_mc.t() * temp1;
            R_cm_Rec = R_cm_Rec / norm(R_cm_Rec); //列向量
            return R_cm_Rec;
        }

        return R_cm_NULL;

    }

    ResultOfMS Tracking::ProjectSN2MF(int a, const cv::Mat &R_mc, const vector<SurfaceNormal> &vTempSurfaceNormal,
                                      vector<FrameLine> &vVanishingDirection, const int numOfSN) {
        vector<cv::Point2d> m_j_selected;
        cv::Mat R_cm_Rec = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        cv::Mat R_cm_NULL = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        ResultOfMS RandDen;
        RandDen.axis = a;

        size_t sizeOfSurfaceNormal = vTempSurfaceNormal.size() + vVanishingDirection.size();
        m_j_selected.reserve(sizeOfSurfaceNormal);

        for (size_t i = 0; i < sizeOfSurfaceNormal; i++) {
            //cv::Mat temp=cv::Mat::zeros(cv::Size(1,3),CV_32F);

            cv::Point3f n_ini;
            int tepSize = i - vTempSurfaceNormal.size();
            if (i >= vTempSurfaceNormal.size()) {

                n_ini.x = R_mc.at<float>(0, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(0, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(0, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.y = R_mc.at<float>(1, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(1, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(1, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.z = R_mc.at<float>(2, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(2, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(2, 2) * vVanishingDirection[tepSize].direction.z;
            } else {

                n_ini.x = R_mc.at<float>(0, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(0, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(0, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.y = R_mc.at<float>(1, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(1, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(1, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.z = R_mc.at<float>(2, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(2, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(2, 2) * vTempSurfaceNormal[i].normal.z;
            }


            double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
            //cout<<lambda<<endl;
            //inside the cone
            if (lambda < sin(0.2518)) //0.25
            {
                double tan_alfa = lambda / std::abs(n_ini.z);
                double alfa = asin(lambda);
                double m_j_x = alfa / tan_alfa * n_ini.x / n_ini.z;
                double m_j_y = alfa / tan_alfa * n_ini.y / n_ini.z;
                if (!std::isnan(m_j_x) && !std::isnan(m_j_y))
                    m_j_selected.push_back(cv::Point2d(m_j_x, m_j_y));
                if (i < vTempSurfaceNormal.size()) {
                    if (a == 1) {
                        mCurrentFrame.vSurfaceNormalx.push_back(vTempSurfaceNormal[i].FramePosition);
                        mCurrentFrame.vSurfacePointx.push_back(vTempSurfaceNormal[i].cameraPosition);
                    } else if (a == 2) {
                        mCurrentFrame.vSurfaceNormaly.push_back(vTempSurfaceNormal[i].FramePosition);
                        mCurrentFrame.vSurfacePointy.push_back(vTempSurfaceNormal[i].cameraPosition);
                    } else if (a == 3) {
                        mCurrentFrame.vSurfaceNormalz.push_back(vTempSurfaceNormal[i].FramePosition);
                        mCurrentFrame.vSurfacePointz.push_back(vTempSurfaceNormal[i].cameraPosition);
                    }
                } else {
                    if (a == 1) {
                        cv::Point2d endPoint = vVanishingDirection[tepSize].p;
                        cv::Point2d startPoint = vVanishingDirection[tepSize].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinex.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[tepSize].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCx.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                    } else if (a == 2) {
                        cv::Point2d endPoint = vVanishingDirection[tepSize].p;
                        cv::Point2d startPoint = vVanishingDirection[tepSize].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLiney.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[tepSize].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCy.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                    } else if (a == 3) {
                        cv::Point2d endPoint = vVanishingDirection[tepSize].p;
                        cv::Point2d startPoint = vVanishingDirection[tepSize].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinez.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[tepSize].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCz.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                    }
                }


            }
        }
        //cout<<"a=1:"<<mCurrentFrame.vSurfaceNormalx.size()<<",a =2:"<<mCurrentFrame.vSurfaceNormaly.size()<<", a=3:"<<mCurrentFrame.vSurfaceNormalz.size()<<endl;
        //cout<<"m_j_selected.push_back(temp)"<<m_j_selected.size()<<endl;

        if (m_j_selected.size() > numOfSN) {
            sMS tempMeanShift = MeanShift(m_j_selected);
            cv::Point2d s_j = tempMeanShift.centerOfShift;// MeanShift(m_j_selected);
            float s_j_density = tempMeanShift.density;
            //cout<<"tracking:s_j"<<s_j.x<<","<<s_j.y<<endl;
            float alfa = norm(s_j);
            float ma_x = tan(alfa) / alfa * s_j.x;
            float ma_y = tan(alfa) / alfa * s_j.y;
            cv::Mat temp1 = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
            temp1.at<float>(0, 0) = ma_x;
            temp1.at<float>(1, 0) = ma_y;
            temp1.at<float>(2, 0) = 1;
            cv::Mat rtemp = R_mc.t();
            R_cm_Rec = rtemp * temp1;
            R_cm_Rec = R_cm_Rec / norm(R_cm_Rec); //列向量
            RandDen.R_cm_Rec = R_cm_Rec;
            RandDen.s_j_density = s_j_density;

            return RandDen;
        }
        RandDen.R_cm_Rec = R_cm_NULL;
        return RandDen;

    }

    axiSNV Tracking::ProjectSN2Conic(int a, const cv::Mat &R_mc, const vector<SurfaceNormal> &vTempSurfaceNormal,
                                     vector<FrameLine> &vVanishingDirection) {
        int numInConic = 0;
        vector<cv::Point2d> m_j_selected;
        cv::Mat R_cm_Rec = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
        cv::Mat R_cm_NULL = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        //cv::Mat R_mc=cv::Mat::zeros(cv::Size(3,3),CV_32F);
        vector<SurfaceNormal> vSNCadidate;
        axiSNV tempaxiSNV;
        tempaxiSNV.axis = a;


        size_t sizeOfSurfaceNormal = vTempSurfaceNormal.size() + vVanishingDirection.size();
        tempaxiSNV.SNVector.reserve(sizeOfSurfaceNormal);
//        cout << "size of SN" << sizeOfSurfaceNormal << endl;
        for (size_t i = 0; i < sizeOfSurfaceNormal; i++) {

            cv::Point3f n_ini;
            if (i < vTempSurfaceNormal.size()) {
                n_ini.x = R_mc.at<float>(0, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(0, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(0, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.y = R_mc.at<float>(1, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(1, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(1, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.z = R_mc.at<float>(2, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(2, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(2, 2) * vTempSurfaceNormal[i].normal.z;

                double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
                //cout<<lambda<<endl;
                if (lambda < sin(0.2018)) //0.25
                {

                    //vSNCadidate.push_back(vTempSurfaceNormal[i]);
                    //numInConic++;
                    tempaxiSNV.SNVector.push_back(vTempSurfaceNormal[i]);


                }
            } else {   //cout<<"vanishing"<<endl;
                int tepSize = i - vTempSurfaceNormal.size();
                //cout<<vVanishingDirection[tepSize].direction.x<<"vanishing"<<endl;

                n_ini.x = R_mc.at<float>(0, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(0, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(0, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.y = R_mc.at<float>(1, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(1, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(1, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.z = R_mc.at<float>(2, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(2, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(2, 2) * vVanishingDirection[tepSize].direction.z;

                double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
                //cout<<lambda<<endl;
                if (lambda < sin(0.1018)) //0.25
                {

                    //vSNCadidate.push_back(vTempSurfaceNormal[i]);
                    //numInConic++;
                    tempaxiSNV.Linesvector.push_back(vVanishingDirection[tepSize]);


                }

            }


        }

        return tempaxiSNV;//numInConic;

    }

    cv::Mat Tracking::TrackManhattanFrame(cv::Mat &mLastRcm, vector<SurfaceNormal> &vSurfaceNormal,
                                          vector<FrameLine> &vVanishingDirection) {
        //cout << "begin Tracking" << endl;
        cv::Mat R_cm_update = mLastRcm.clone();
        int isTracked = 0;
        vector<double> denTemp(3, 0.00001);
        for (int i = 0; i <1; i++) {

            cv::Mat R_cm = R_cm_update;//cv::Mat::eye(cv::Size(3,3),CV_32FC1);  // 对角线为1的对角矩阵(3, 3, CV_32FC1);
            int directionFound1 = 0;
            int directionFound2 = 0;
            int directionFound3 = 0; //三个方向
            int numDirectionFound = 0;
            vector<axiSNVector> vaxiSNV(4);
            vector<int> numInCone = vector<int>(3, 0);
            vector<cv::Point2f> vDensity;
            //chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            for (int a = 1; a < 4; a++) {
                //在每个conic有多少 点
                cv::Mat R_mc = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                int c1 = (a + 3) % 3;
                int c2 = (a + 4) % 3;
                int c3 = (a + 5) % 3;
                R_mc.at<float>(0, 0) = R_cm.at<float>(0, c1);
                R_mc.at<float>(0, 1) = R_cm.at<float>(0, c2);
                R_mc.at<float>(0, 2) = R_cm.at<float>(0, c3);
                R_mc.at<float>(1, 0) = R_cm.at<float>(1, c1);
                R_mc.at<float>(1, 1) = R_cm.at<float>(1, c2);
                R_mc.at<float>(1, 2) = R_cm.at<float>(1, c3);
                R_mc.at<float>(2, 0) = R_cm.at<float>(2, c1);
                R_mc.at<float>(2, 1) = R_cm.at<float>(2, c2);
                R_mc.at<float>(2, 2) = R_cm.at<float>(2, c3);
                cv::Mat R_mc_new = R_mc.t();
//                cout << "R_mc_new" << R_mc_new << endl;
                vaxiSNV[a - 1] = ProjectSN2Conic(a, R_mc_new, vSurfaceNormal, vVanishingDirection);
                numInCone[a - 1] = vaxiSNV[a - 1].SNVector.size();
                //cout<<"2 a:"<<vaxiSNV[a-1].axis<<",vector:"<<numInCone[a - 1]<<endl;
            }
            //chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
            //cout << "first sN time: " << time_used.count() << endl;
            int minNumOfSN = vSurfaceNormal.size() / 20;
            //cout<<"minNumOfSN"<<minNumOfSN<<endl;
            //排序  a<b<c
            int a = numInCone[0];
            int b = numInCone[1];
            int c = numInCone[2];
            //cout<<"a:"<<a<<",b:"<<b<<",c:"<<c<<endl;
            //a,b,c升序排列
            int temp = 0;
            if (a > b) temp = a, a = b, b = temp;
            if (b > c) temp = b, b = c, c = temp;
            if (a > b) temp = a, a = b, b = temp;
            //cout<<"sequence  a:"<<a<<",b:"<<b<<",c:"<<c<<endl;
            if (b < minNumOfSN) {
                minNumOfSN = (b + a) / 2;
                cout << "thr " << minNumOfSN << endl;
            }

            //cout<<"new  minNumOfSN"<<minNumOfSN<<endl;
            //chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
            for (int a = 1; a < 4; a++) {
                cv::Mat R_mc = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                int c1 = (a + 3) % 3;
                int c2 = (a + 4) % 3;
                int c3 = (a + 5) % 3;
                R_mc.at<float>(0, 0) = R_cm.at<float>(0, c1);
                R_mc.at<float>(0, 1) = R_cm.at<float>(0, c2);
                R_mc.at<float>(0, 2) = R_cm.at<float>(0, c3);
                R_mc.at<float>(1, 0) = R_cm.at<float>(1, c1);
                R_mc.at<float>(1, 1) = R_cm.at<float>(1, c2);
                R_mc.at<float>(1, 2) = R_cm.at<float>(1, c3);
                R_mc.at<float>(2, 0) = R_cm.at<float>(2, c1);
                R_mc.at<float>(2, 1) = R_cm.at<float>(2, c2);
                R_mc.at<float>(2, 2) = R_cm.at<float>(2, c3);
                cv::Mat R_mc_new = R_mc.t();
                vector<SurfaceNormal> *tempVVSN;
                vector<FrameLine> *tempLineDirection;
                for (int i = 0; i < 3; i++) {
                    if (vaxiSNV[i].axis == a) {

                        tempVVSN = &vaxiSNV[i].SNVector;
                        tempLineDirection = &vaxiSNV[i].Linesvector;
                        break;
                    }

                }

                ResultOfMS RD_temp = ProjectSN2MF(a, R_mc_new, *tempVVSN, *tempLineDirection, minNumOfSN);

                //chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
                //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t4-t3);
                //cout << "second SN time: " << time_used.count() << endl;

                //cout << "test projectSN2MF" << ra << endl;
                //sum求矩阵元素和  如果是单通道时需要加[0]
                if (sum(RD_temp.R_cm_Rec)[0] != 0) {
                    numDirectionFound += 1;
                    if (a == 1) directionFound1 = 1;//第一个轴
                    else if (a == 2) directionFound2 = 1;
                    else if (a == 3) directionFound3 = 1;
                    R_cm_update.at<float>(0, a - 1) = RD_temp.R_cm_Rec.at<float>(0, 0);
                    R_cm_update.at<float>(1, a - 1) = RD_temp.R_cm_Rec.at<float>(1, 0);
                    R_cm_update.at<float>(2, a - 1) = RD_temp.R_cm_Rec.at<float>(2, 0);
                    //RD_temp.s_j_density;

                    vDensity.push_back(cv::Point2f(RD_temp.axis, RD_temp.s_j_density));

                }
            }

            if (numDirectionFound < 2) {
                cout << "oh, it has happened" << endl;
                R_cm_update = R_cm;
                numDirectionFound = 0;
                isTracked = 0;
                directionFound1 = 0;
                directionFound2 = 0;
                directionFound3 = 0;
                break;
            } else if (numDirectionFound == 2) {
                //通过两个正交的轴计算出第三个轴
                if (directionFound1 && directionFound2) {
                    cv::Mat v1 = R_cm_update.colRange(0, 1).clone();
                    cv::Mat v2 = R_cm_update.colRange(1, 2).clone();
                    cv::Mat v3 = v1.cross(v2);
                    R_cm_update.at<float>(0, 2) = v3.at<float>(0, 0);
                    R_cm_update.at<float>(1, 2) = v3.at<float>(1, 0);
                    R_cm_update.at<float>(2, 2) = v3.at<float>(2, 0);
                    //determinant计算行列式
                    if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                        R_cm_update.at<float>(0, 2) = -v3.at<float>(0, 0);
                        R_cm_update.at<float>(1, 2) = -v3.at<float>(1, 0);
                        R_cm_update.at<float>(2, 2) = -v3.at<float>(2, 0);
                    }

                } else if (directionFound2 && directionFound3) {
                    cv::Mat v2 = R_cm_update.colRange(1, 2).clone();
                    cv::Mat v3 = R_cm_update.colRange(2, 3).clone();
                    cv::Mat v1 = v3.cross(v2);
                    R_cm_update.at<float>(0, 0) = v1.at<float>(0, 0);
                    R_cm_update.at<float>(1, 0) = v1.at<float>(1, 0);
                    R_cm_update.at<float>(2, 0) = v1.at<float>(2, 0);
                    if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                        R_cm_update.at<float>(0, 0) = -v1.at<float>(0, 0);
                        R_cm_update.at<float>(1, 0) = -v1.at<float>(1, 0);
                        R_cm_update.at<float>(2, 0) = -v1.at<float>(2, 0);
                    }
                } else if (directionFound1 && directionFound3) {
                    cv::Mat v1 = R_cm_update.colRange(0, 1).clone();
                    cv::Mat v3 = R_cm_update.colRange(2, 3).clone();
                    cv::Mat v2 = v1.cross(v3);
                    R_cm_update.at<float>(0, 1) = v2.at<float>(0, 0);
                    R_cm_update.at<float>(1, 1) = v2.at<float>(1, 0);
                    R_cm_update.at<float>(2, 1) = v2.at<float>(2, 0);
                    if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                        R_cm_update.at<float>(0, 1) = -v2.at<float>(0, 0);
                        R_cm_update.at<float>(1, 1) = -v2.at<float>(1, 0);
                        R_cm_update.at<float>(2, 1) = -v2.at<float>(2, 0);
                    }

                }
            }
            //cout<<"svd before"<<R_cm_update<<endl;
            SVD svd;
            cv::Mat U, W, VT;

            svd.compute(R_cm_update, W, U, VT);

            R_cm_update = U* VT;
            vDensity.clear();
            //acos()计算余弦值,trace()计算对角元素和
            if (acos((trace(R_cm.t() * R_cm_update)[0] - 1.0)) / 2 < 0.001) {
                cout << "go outside" << endl;
                break;
            }
        }
        isTracked = 1;
        return R_cm_update.clone();
    }

    sMS Tracking::MeanShift(vector<cv::Point2d> &v2D) {
        sMS tempMS;
        int numPoint = v2D.size();
        float density;
        cv::Point2d nominator;
        double denominator = 0;
        double nominator_x = 0;
        double nominator_y = 0;
        for (int i = 0; i < numPoint; i++) {
            double k = exp(-20 * norm(v2D.at(i)) * norm(v2D.at(i)));
            nominator.x += k * v2D.at(i).x;
            nominator.y += k * v2D.at(i).y;
            denominator += k;
        }
        tempMS.centerOfShift = nominator / denominator;
        tempMS.density = denominator / numPoint;

        return tempMS;
    }

    void Tracking::StereoInitialization() {
        if (mCurrentFrame.N > 50 || mCurrentFrame.NL > 15) {
            // Set Frame pose to the origin
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

            // Create KeyFrame
            // 将当前帧构造为初始关键帧
            // mCurrentFrame的数据类型为Frame
            // KeyFrame包含Frame、地图3D点、以及BoW
            // KeyFrame里有一个mpMap，Tracking里有一个mpMap，而KeyFrame里的mpMap都指向Tracking里的这个mpMap
            // KeyFrame里有一个mpKeyFrameDB，Tracking里有一个mpKeyFrameDB，而KeyFrame里的mpMap都指向Tracking里的这个mpKeyFrameDB
            // 提问: 为什么要指向Tracking中的相应的变量呢? -- 因为Tracking是主线程，是它创建和加载的这些模块
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

            // Insert KeyFrame in the map
            // KeyFrame中包含了地图、反过来地图中也包含了KeyFrame，相互包含
            // 在地图中添加该初始关键帧
            mpMap->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            // 为每个特征点构造MapPoint
            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    // 通过反投影得到该特征点的世界坐标系下3D坐标
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    // 将3D点构造为MapPoint
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);

                    // 为该MapPoint添加属性：
                    // a.观测到该MapPoint的关键帧
                    // b.该MapPoint的描述子
                    // c.该MapPoint的平均观测方向和深度范围

                    // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
                    pNewMP->AddObservation(pKFini, i);
                    // 表示该KeyFrame的哪个特征点可以观测到哪个3D点
                    pKFini->AddMapPoint(pNewMP, i);
                    // b.从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
                    pNewMP->ComputeDistinctiveDescriptors();
                    // c.更新该MapPoint平均观测方向以及观测距离的范围
                    pNewMP->UpdateNormalAndDepth();

                    // 在地图中添加该MapPoint
                    mpMap->AddMapPoint(pNewMP);

                    // 将该MapPoint添加到当前帧的mvpMapPoints中
                    // 为当前Frame的特征点与MapPoint之间建立索引
                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }


            for (int i = 0; i < mCurrentFrame.NL; i++) {

                float z = mCurrentFrame.mvDepthLine[i];

                if (z > 0) {
                    Vector6d line3D = mCurrentFrame.obtain3DLine(i);
                    // 将3D线构造为MapPoint
                    MapLine *pNewML = new MapLine(line3D, pKFini, mpMap);
                    pNewML->AddObservation(pKFini, i);
                    pKFini->AddMapLine(pNewML, i);
                    pNewML->ComputeDistinctiveDescriptors();
                    pNewML->UpdateAverageDir();
                    mpMap->AddMapLine(pNewML);
                    mCurrentFrame.mvpMapLines[i] = pNewML;
                }
            }

            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                MapPlane *pNewMP = new MapPlane(p3D, pKFini, mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPlane(pNewMP, i);
                pNewMP->UpdateCoefficientsAndPoints();
                mpMap->AddMapPlane(pNewMP);
                mCurrentFrame.mvpMapPlanes[i] = pNewMP;
cout<<"i   "<<i<<endl;
cout<<"mnPlaneNum   "<<mCurrentFrame.mnPlaneNum<<endl;
cout<<"mCurrentFrame.mvpMapPlanes[i] = pNewMP   "<<pNewMP<<endl;
            }

            mpPointCloudMapping->print();

            // 在局部地图中添加该初始关键帧
            mpLocalMapper->InsertKeyFrame(pKFini);

            // 更新当前帧为上一帧
            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpMap->GetAllMapPoints();
            mvpLocalMapLines = mpMap->GetAllMapLines();

            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
            // ReferenceMapPoints是DrawMapPoints函数画图的时候用的
            // Lines同理
            mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
            mpMap->SetReferenceMapLines(mvpLocalMapLines);

            mpMap->mvpKeyFrameOrigins.push_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;
        }
    }

    void Tracking::MonocularInitialization() {
        int num = 100;
        // 如果单目初始器还没有没创建，则创建单目初始器
        if (!mpInitializer) {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > num) {
                // step 1：得到用于初始化的第一帧，初始化需要两帧
                mInitialFrame = Frame(mCurrentFrame);
                // 记录最近的一帧
                mLastFrame = Frame(mCurrentFrame);
                // mvbPreMatched最大的情况就是当前帧所有的特征点都被匹配上
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                if (mpInitializer)
                    delete mpInitializer;

                // 由当前帧构造初始化器， sigma:1.0    iterations:200
                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                return;
            }
        } else {
            // Try to initialize
            if ((int) mCurrentFrame.mvKeys.size() <= num) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }

            // Find correspondences
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches,
                                                           100);

            LSDmatcher lmatcher;   //建立线特征之间的匹配
            int lineMatches = lmatcher.SerachForInitialize(mInitialFrame, mCurrentFrame, mvLineMatches);

            if (nmatches < 100) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                return;
            }

            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)


#if 0
                                                                                                                                    if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
        for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
        {
            if(mvIniMatches[i]>=0 && !vbTriangulated[i])
            {
                mvIniMatches[i]=-1;
                nmatches--;
            }
        }

        // Set Frame Poses
        // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
        mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
        cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(Tcw.rowRange(0,3).col(3));
        mCurrentFrame.SetPose(Tcw);

        // step6：将三角化得到的3D点包装成MapPoints
        /// 如果要修改，应该是从这个函数开始
        CreateInitialMapMonocular();
    }
#else
            if (0)//mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated, mvLineMatches, mvLineS3D, mvLineE3D, mvbLineTriangulated))
            {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

                // step6：将三角化得到的3D点包装成MapPoints
                /// 如果要修改，应该是从这个函数开始
//            CreateInitialMapMonocular();
                CreateInitialMapMonoWithLine();
            }
#endif
        }
    }

    void Tracking::CreateInitialMapMonocular() {
        // Create KeyFrames
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // Create MapPoints and asscoiate to keyframes
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);

            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // b.从众多观测到该MapPoint的特征点中挑选出区分度最高的描述子
            pMP->ComputeDistinctiveDescriptors();

            // c.更新该MapPoint的平均观测方向以及观测距离的范围
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            //Add to Map
            mpMap->AddMapPoint(pMP);
        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
            cout << "Wrong initialization, reseting..." << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;  //至此，初始化成功
    }

#if 1

/**
* @brief 为单目摄像头三角化生成带有线特征的Map，包括MapPoints和MapLine
*/
    void Tracking::CreateInitialMapMonoWithLine() {
        // step1:
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // step2：
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // step3：
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // step4：
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            // Create MapPoint
            cv::Mat worldPos(mvIniP3D[i]);

            // step4.1：
            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            // step4.2：

            // step4.3：
            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            // a.
            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // b.
            pMP->UpdateNormalAndDepth();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            // Add to Map
            // step4.4：
            mpMap->AddMapPoint(pMP);
        }

        // step5：
        for (size_t i = 0; i < mvLineMatches.size(); i++) {
            if (!mvbLineTriangulated[i])
                continue;

            // Create MapLine
            Vector6d worldPos;
            worldPos << mvLineS3D[i].x, mvLineS3D[i].y, mvLineS3D[i].z, mvLineE3D[i].x, mvLineE3D[i].y, mvLineE3D[i].z;

            //step5.1：
            MapLine *pML = new MapLine(worldPos, pKFcur, mpMap);

            pKFini->AddMapLine(pML, i);
            pKFcur->AddMapLine(pML, i);

            //a.
            pML->AddObservation(pKFini, i);
            pML->AddObservation(pKFcur, i);

            //b.
            pML->ComputeDistinctiveDescriptors();

            //c.
            pML->UpdateAverageDir();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapLines[i] = pML;
            mCurrentFrame.mvbLineOutlier[i] = false;

            // step5.4: Add to Map
            mpMap->AddMapLine(pML);
        }


        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

//        cout << "medianDepth = " << medianDepth << endl;
//        cout << "pKFcur->TrackedMapPoints(1) = " << pKFcur->TrackedMapPoints(1) << endl;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 80) {
            cout << "Wrong initialization, reseting ... " << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale Points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); ++iMP) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        // Scale Line Segments
        vector<MapLine *> vpAllMapLines = pKFini->GetMapLineMatches();
        for (size_t iML = 0; iML < vpAllMapLines.size(); iML++) {
            if (vpAllMapLines[iML]) {
                MapLine *pML = vpAllMapLines[iML];
                pML->SetWorldPos(pML->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mvpLocalMapLines = mpMap->GetAllMapLines();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;
    }

#endif

    void Tracking::CheckReplacedInLastFrame() {
        for (int i = 0; i < mLastFrame.N; i++) {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (pMP) {
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }

        for (int i = 0; i < mLastFrame.NL; i++) {
            MapLine *pML = mLastFrame.mvpMapLines[i];

            if (pML) {
                MapLine *pReL = pML->GetReplaced();
                if (pReL) {
                    mLastFrame.mvpMapLines[i] = pReL;
                }
            }
        }

        for (int i = 0; i < mLastFrame.mnPlaneNum; i++) {
            MapPlane *pMP = mLastFrame.mvpMapPlanes[i];

            if (pMP) {
                MapPlane *pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPlanes[i] = pRep;
                }
            }
        }
    }

    bool Tracking::DetectManhattan() {
        KeyFrame *pKFCandidate = nullptr;
        int maxScore = 0;
        cv::Mat pMFc1, pMFc2, pMFc3, pMFm1, pMFm2, pMFm3;
        fullManhattanFound = false;

        int id1, id2, id3 = -1;

        for (size_t i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            cv::Mat p3Dc1 = mCurrentFrame.mvPlaneCoefficients[i];
            MapPlane *pMP1 = mCurrentFrame.mvpMapPlanes[i];
cout<<"i "<<i<<endl;
cout<<"pMP1 "<<pMP1<<endl;
cout<<"mCurrentFrame.mvpMapPlanes[i] "<<mCurrentFrame.mvpMapPlanes[0]<<endl;
            if (!pMP1 || pMP1->isBad()) {
                continue;
            }

            for (size_t j = i + 1; j < mCurrentFrame.mnPlaneNum; j++) {
                cv::Mat p3Dc2 = mCurrentFrame.mvPlaneCoefficients[j];
                MapPlane *pMP2 = mCurrentFrame.mvpMapPlanes[j];

                if (!pMP2 || pMP2->isBad()) {
                    continue;
                }

                float angle12 = p3Dc1.at<float>(0) * p3Dc2.at<float>(0) +
                                p3Dc1.at<float>(1) * p3Dc2.at<float>(1) +
                                p3Dc1.at<float>(2) * p3Dc2.at<float>(2);

                if (angle12 > mfMFVerTh || angle12 < -mfMFVerTh) {
                    continue;
                }

                for (size_t k = j + 1; k < mCurrentFrame.mnPlaneNum; k++) {
                    cv::Mat p3Dc3 = mCurrentFrame.mvPlaneCoefficients[k];
                    MapPlane *pMP3 = mCurrentFrame.mvpMapPlanes[k];

                    if (!pMP3 || pMP3->isBad()) {
                        continue;
                    }

                    float angle13 = p3Dc1.at<float>(0) * p3Dc3.at<float>(0) +
                                    p3Dc1.at<float>(1) * p3Dc3.at<float>(1) +
                                    p3Dc1.at<float>(2) * p3Dc3.at<float>(2);

                    float angle23 = p3Dc2.at<float>(0) * p3Dc3.at<float>(0) +
                                    p3Dc2.at<float>(1) * p3Dc3.at<float>(1) +
                                    p3Dc2.at<float>(2) * p3Dc3.at<float>(2);

                    if (angle13 > mfMFVerTh || angle13 < -mfMFVerTh || angle23 > mfMFVerTh || angle23 < -mfMFVerTh) {
                        continue;
                    }

                    KeyFrame *pKF = mpMap->GetManhattanObservation(pMP1, pMP2, pMP3);

                    if (!pKF) {
                        continue;
                    }

                    auto idx1 = pMP1->GetIndexInKeyFrame(pKF);
                    auto idx2 = pMP2->GetIndexInKeyFrame(pKF);
                    auto idx3 = pMP3->GetIndexInKeyFrame(pKF);

                    if (idx1 == -1 || idx2 == -1 || idx3 == -1) {
                        continue;
                    }

                    int score = pKF->mvPlanePoints[idx1].size() +
                                pKF->mvPlanePoints[idx2].size() +
                                pKF->mvPlanePoints[idx3].size() +
                                mCurrentFrame.mvPlanePoints[i].size() +
                                mCurrentFrame.mvPlanePoints[j].size() +
                                mCurrentFrame.mvPlanePoints[k].size();

                    if (score > maxScore) {
                        maxScore = score;

                        pKFCandidate = pKF;
                        pMFc1 = p3Dc1;
                        pMFc2 = p3Dc2;
                        pMFc3 = p3Dc3;
                        pMFm1 = pKF->mvPlaneCoefficients[idx1];
                        pMFm2 = pKF->mvPlaneCoefficients[idx2];
                        pMFm3 = pKF->mvPlaneCoefficients[idx3];

                        id1 = pMP1->mnId;
                        id2 = pMP2->mnId;
                        id3 = pMP3->mnId;

                        fullManhattanFound = true;
                    }
                }

                KeyFrame *pKF = mpMap->GetPartialManhattanObservation(pMP1, pMP2);

                if (!pKF) {
                    continue;
                }

                auto idx1 = pMP1->GetIndexInKeyFrame(pKF);
                auto idx2 = pMP2->GetIndexInKeyFrame(pKF);

                if (idx1 == -1 || idx2 == -1) {
                    continue;
                }

                int score = pKF->mvPlanePoints[idx1].size() +
                            pKF->mvPlanePoints[idx2].size() +
                            mCurrentFrame.mvPlanePoints[i].size() +
                            mCurrentFrame.mvPlanePoints[j].size();

                if (score > maxScore) {
                    maxScore = score;

                    pKFCandidate = pKF;
                    pMFc1 = p3Dc1;
                    pMFc2 = p3Dc2;
                    pMFm1 = pKF->mvPlaneCoefficients[idx1];
                    pMFm2 = pKF->mvPlaneCoefficients[idx2];

                    id1 = pMP1->mnId;
                    id2 = pMP2->mnId;

                    fullManhattanFound = false;
                }
            }
        }

        if (pKFCandidate == nullptr) {
            return false;
        }

        if (!fullManhattanFound) {
            cv::Mat pMFc1n = (cv::Mat_<float>(3, 1) << pMFc1.at<float>(0), pMFc1.at<float>(1), pMFc1.at<float>(2));
            cv::Mat pMFc2n = (cv::Mat_<float>(3, 1) << pMFc2.at<float>(0), pMFc2.at<float>(1), pMFc2.at<float>(2));
            pMFc3 = pMFc1n.cross(pMFc2n);

            cv::Mat pMFm1n = (cv::Mat_<float>(3, 1) << pMFm1.at<float>(0), pMFm1.at<float>(1), pMFm1.at<float>(2));
            cv::Mat pMFm2n = (cv::Mat_<float>(3, 1) << pMFm2.at<float>(0), pMFm2.at<float>(1), pMFm2.at<float>(2));
            pMFm3 = pMFm1n.cross(pMFm2n);
        }

        cv::Mat MFc, MFm;
        MFc = cv::Mat::eye(cv::Size(3, 3), CV_32F);
        MFm = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        MFc.at<float>(0, 0) = pMFc1.at<float>(0);
        MFc.at<float>(1, 0) = pMFc1.at<float>(1);
        MFc.at<float>(2, 0) = pMFc1.at<float>(2);
        MFc.at<float>(0, 1) = pMFc2.at<float>(0);
        MFc.at<float>(1, 1) = pMFc2.at<float>(1);
        MFc.at<float>(2, 1) = pMFc2.at<float>(2);
        MFc.at<float>(0, 2) = pMFc3.at<float>(0);
        MFc.at<float>(1, 2) = pMFc3.at<float>(1);
        MFc.at<float>(2, 2) = pMFc3.at<float>(2);

        if (!fullManhattanFound && std::abs(cv::determinant(MFc) + 1) < 0.5) {
            MFc.at<float>(0, 2) = -pMFc3.at<float>(0);
            MFc.at<float>(1, 2) = -pMFc3.at<float>(1);
            MFc.at<float>(2, 2) = -pMFc3.at<float>(2);
        }

        cv::Mat Uc, Wc, VTc;

        cv::SVD::compute(MFc, Wc, Uc, VTc);

        MFc = Uc * VTc;

        MFm.at<float>(0, 0) = pMFm1.at<float>(0);
        MFm.at<float>(1, 0) = pMFm1.at<float>(1);
        MFm.at<float>(2, 0) = pMFm1.at<float>(2);
        MFm.at<float>(0, 1) = pMFm2.at<float>(0);
        MFm.at<float>(1, 1) = pMFm2.at<float>(1);
        MFm.at<float>(2, 1) = pMFm2.at<float>(2);
        MFm.at<float>(0, 2) = pMFm3.at<float>(0);
        MFm.at<float>(1, 2) = pMFm3.at<float>(1);
        MFm.at<float>(2, 2) = pMFm3.at<float>(2);

        if (!fullManhattanFound && std::abs(cv::determinant(MFm) + 1) < 0.5) {
            MFm.at<float>(0, 2) = -pMFm3.at<float>(0);
            MFm.at<float>(1, 2) = -pMFm3.at<float>(1);
            MFm.at<float>(2, 2) = -pMFm3.at<float>(2);
        }

        cv::Mat Um, Wm, VTm;

        cv::SVD::compute(MFm, Wm, Um, VTm);

        MFm = Um * VTm;

        cv::Mat Rwc = pKFCandidate->GetPoseInverse().rowRange(0, 3).colRange(0, 3) * MFm * MFc.t();
        manhattanRcw = Rwc.t();

        return true;
    }

    bool Tracking::TranslationEstimation() {

        // Compute Bag of Words vector
        //将当前帧的描述子转化为BoW向量
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        vector<MapPoint *> vpMapPointMatches;
        vector<MapLine *> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        // 当前帧和参考关键帧用BoW进行快速匹配，匹配结果记录在vpMapPointMatches，nmatches表示匹配的数目
        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
//        vpMapLineMatches = vector<MapLine *>(mCurrentFrame.NL, static_cast<MapLine *>(NULL));
//        int lmatches = 0;

        // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些
        mCurrentFrame.SetPose(mLastFrame.mTcw);

        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());
//        int planeMatches = 0;

        int initialMatches = nmatches + lmatches + planeMatches;

//        cout << "TranslationEstimation: Before: Point matches: " << nmatches << " , Line Matches:"
//             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
            cout << "******************TranslationEstimation: Before: Not enough matches" << endl;
            return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.mvpMapLines = vpMapLineMatches;

        //优化误差函数获取位姿
        //cout << "translation reference,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::TranslationOptimization(&mCurrentFrame);
        //cout << "translation reference,pose after opti" << mCurrentFrame.mTcw << endl;

        int nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        //剔除优化后的匹配点中的外点
        //之所以在优化之后才剔除外点，是因为在优化的过程中就有了对这些外点的标记
        // Discard outliers
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                //如果对应到的某个特征点是外点
                if (mCurrentFrame.mvbOutlier[i]) {
                    //清除它在当前帧中存在过的痕迹
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;

                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    //匹配的内点计数++
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;

            }
        }

        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if ( finalMatches < 3) {
            cout << "TranslationEstimation: After: Not enough matches" << endl;
            mCurrentFrame.SetPose(mLastFrame.mTcw);
            return false;
        }

        return true;
    }

/**
* @brief 根据恒定速度模型用上一帧地图点来对当前帧进行跟踪。在正常情况下，系统使用恒速模型进行跟踪当前帧。该函数是用上一帧来跟踪当前帧。
* Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
* Step 2：根据上一帧特征点对应地图点进行投影匹配
* Step 3：优化当前帧位姿
* Step 4：剔除地图点中外点
* @return
*/
    bool Tracking::TranslationWithMotionModel() {
        ORBmatcher matcher(0.9, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        //更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
        UpdateLastFrame();

//        根据之前估计的速度，用恒速模型得到当前帧的初始位姿。
        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        // Project points seen in previous frame
        // 设置特征匹配过程中的搜索半径
        int th;
        if (mSensor != System::STEREO)
            th = 15;//单目
        else
            th = 7;//双目

        // 清空当前帧的地图点
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
//        用上一帧地图点进行投影匹配，如果匹配点不够，则扩大搜索半径再来一次
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);
        vector<MapLine *> vpMapLineMatches;
        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        mCurrentFrame.mvpMapLines = vpMapLineMatches;

        // 如果匹配点太少，则扩大搜索半径再来一次
        if (nmatches <50) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.MatchORBPoints(mCurrentFrame, mLastFrame);
            mUpdateMF = true;
        }

        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        int initialMatches = nmatches + lmatches + planeMatches;

        if (initialMatches < 10) {
            cout << "TranslationWithMotionModel: Before: Not enough matches" << endl;
            return false;
        }

        // Optimize frame pose with all matches
        mRotation_wc.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));


        //cout << "translation motion model,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::TranslationOptimization(&mCurrentFrame);
        //cout << "translation motion model,pose after opti" << mCurrentFrame.mTcw << endl;


        // Discard outliers
        int nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;
            }

        }

        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        if (mbOnlyTracking) {
            // 纯定位模式下：如果成功追踪的地图点非常少,那么这里的mbVO标志就会置位
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if (nmatchesMap < 3 ||finalMatches < 3) {
            cout << "TranslationWithMotionModel: After: Not enough matches" << endl;
            mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
            return false;
        }

        return true;
    }

/**
 * @brief 更新上一帧位姿，在上一帧中生成临时地图点
 * 单目情况：只计算了上一帧的世界坐标系位姿
 * 双目和rgbd情况：选取有有深度值的并且没有被选为地图点的点生成新的临时地图点，提高跟踪鲁棒性
 */
    void Tracking::UpdateLastFrame() {
        // Update pose according to reference keyframe
        // 利用参考关键帧更新上一帧在世界坐标系下的位姿
        // 上一普通帧的参考关键帧，注意这里用的是参考关键帧（位姿准）而不是上上一帧的普通帧
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        // 将上一帧的世界坐标系下的位姿计算出来
        // l:last, r:reference, w:world
        // Tlw = Tlr*Trw
        mLastFrame.SetPose(Tlr * pRef->GetPose());

        // 如果上一帧为关键帧，或者单目的情况，以及不是定位模式时则退出
        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
            return;

//        对于双目或rgbd相机，为上一帧生成新的临时地图点
        // 注意这些地图点只是用来跟踪，不加入到地图中，跟踪完后会删除

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for (int i = 0; i < mLastFrame.N; i++) {
            float z = mLastFrame.mvDepth[i];
            if (z > 0) {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++) {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1) {
                bCreateNew = true;
            }

            if (bCreateNew) {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }


        // Create "visual odometry" MapLines
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vLineDepthIdx;
        vLineDepthIdx.reserve(mLastFrame.NL);
        int nLines = 0;
        for (int i = 0; i < mLastFrame.NL; i++) {
            float z = mLastFrame.mvDepthLine[i];
            if (z == 1) {
                bool bCreateNew = false;
                vLineDepthIdx.push_back(make_pair(z, i));
                MapLine *pML = mLastFrame.mvpMapLines[i];
                if (!pML)
                    bCreateNew = true;
                else if (pML->Observations() < 1) {
                    bCreateNew = true;
                }
                if (bCreateNew) {
                    Vector6d line3D = mLastFrame.obtain3DLine(i);//mvLines3D[i];
                    MapLine *pNewML = new MapLine(line3D, mpMap, &mLastFrame, i);
                    //Vector6d x3D = mLastFrame.mvLines3D(i);
                    //MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);
                    mLastFrame.mvpMapLines[i] = pNewML;

                    mlpTemporalLines.push_back(pNewML);
                    nLines++;
                } else {
                    nLines++;
                }

                if (nLines > 30)
                    break;

            }
        }


    }

/**
 * @brief 用局部地图进行跟踪，进一步优化位姿
 *
 * 1. 更新局部地图，包括局部关键帧和关键点
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 *
 * Step 1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
 * Step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
 * Step 3：更新局部所有MapPoints后对位姿再次优化
 * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
 * Step 5：决定是否跟踪成功
 */
    bool Tracking::TrackLocalMap()  {
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        UpdateLocalMap();

        thread threadPoints(&Tracking::SearchLocalPoints, this);
        thread threadLines(&Tracking::SearchLocalLines, this);
        thread threadPlanes(&Tracking::SearchLocalPlanes, this);
        threadPoints.join();
        threadLines.join();
        threadPlanes.join();

        pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes());

        //cout << "tracking localmap with lines, pose before opti" << endl << mCurrentFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame);
//        Optimizer::TranslationOptimization(&mCurrentFrame);
        //cout << "tracking localmap with lines, pose after opti" << mCurrentFrame.mTcw << endl;

        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (!mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                } else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);

            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (!mCurrentFrame.mvbLineOutlier[i]) {
                    mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                } else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
            }
        }

        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
//                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
//                    mCurrentFrame.mvbPlaneOutlier[i]=false;
//                    nDiscardPlane++;
                } else {
                    mCurrentFrame.mvpMapPlanes[i]->IncreaseFound();
                    mnMatchesInliers++;
                }
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 7) {
            cout << "TrackLocalMapWithLines: After: Not enough matches" << endl;
            return false;
        }


        if (mnMatchesInliers < 7) {
            cout << "TrackLocalMapWithLines: After: Not enough matches" << endl;
            return false;
        } else
            return true;
    }


    bool Tracking::NeedNewKeyFrame() {
        if (mbOnlyTracking)
            return false;

// If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;

        const int nKFs = mpMap->KeyFramesInMap();

// Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

// Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

// Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

// Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
// "total matches = matches to map + visual odometry matches"
// Visual odometry matches will become MapPoints if we insert a keyframe.
// This ratio measures how many MapPoints we could create if we insert a keyframe.
        int nMap = 0; //nTrackedClose
        int nTotal = 0;
        int nNonTrackedClose = 0;
        if (mSensor != System::MONOCULAR) {
            for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                    nTotal++;
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nMap++;
                    else
                        nNonTrackedClose++;
                }
            }
        } else {
            // There are no visual odometry matches in the monocular case
            nMap = 1;
            nTotal = 1;
        }

        const float ratioMap = (float) nMap / fmax(1.0f, nTotal);

// Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        float thMapRatio = 0.35f;
        if (mnMatchesInliers > 300)
            thMapRatio = 0.20f;

// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
//Condition 1c: tracking is weak
        const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || ratioMap < 0.3f);
// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || ratioMap < thMapRatio) &&
                         mnMatchesInliers > 15);

        if (((c1a || c1b || c1c) && c2) || mCurrentFrame.mbNewPlane) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle) {
                return true;
            } else {
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR) {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                        return true;
                    else
                        return false;
                } else
                    return false;
            }
        }

        return false;
    }

    void Tracking::CreateNewKeyFrame() {
        if (!mpLocalMapper->SetNotStop(true))
            return;

        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        if (mSensor != System::MONOCULAR) {

            mCurrentFrame.UpdatePoseMatrices();
            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            vector<pair<float, int> > vDepthIdx;
            vDepthIdx.reserve(mCurrentFrame.N);

            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty()) {
                sort(vDepthIdx.begin(), vDepthIdx.end());

                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size(); j++) {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }

                    if (bCreateNew) {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                        pNewMP->AddObservation(pKF, i);
                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpMap->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    } else {
                        nPoints++;
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                        break;
                }
            }

            vector<pair<float, int>> vLineDepthIdx;
            vLineDepthIdx.reserve(mCurrentFrame.NL);

            for (int i = 0; i < mCurrentFrame.NL; i++) {
                float z = mCurrentFrame.mvDepthLine[i];
                if (z > 0) {
                    vLineDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vLineDepthIdx.empty()) {
                sort(vLineDepthIdx.begin(),vLineDepthIdx.end());

                int nLines = 0;
                for (size_t j = 0; j < vLineDepthIdx.size(); j++) {
                    int i = vLineDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapLine *pMP = mCurrentFrame.mvpMapLines[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    }

                    if (bCreateNew) {
                        Vector6d line3D = mCurrentFrame.obtain3DLine(i);//mvLines3D[i];
                        MapLine *pNewML = new MapLine(line3D, pKF, mpMap);
                        pNewML->AddObservation(pKF, i);
                        pKF->AddMapLine(pNewML, i);
                        pNewML->ComputeDistinctiveDescriptors();
                        pNewML->UpdateAverageDir();
                        mpMap->AddMapLine(pNewML);
                        mCurrentFrame.mvpMapLines[i] = pNewML;
                        nLines++;
                    } else {
                        nLines++;
                    }

//                    if (nLines > 20)
                    if (nLines > 50)
                        break;
                }
            }

            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                if (mCurrentFrame.mvpParallelPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i]->AddParObservation(pKF, i);
                }
                if (mCurrentFrame.mvpVerticalPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i]->AddVerObservation(pKF, i);
                }

                if (mCurrentFrame.mvpMapPlanes[i]) {
                    mCurrentFrame.mvpMapPlanes[i]->AddObservation(pKF, i);
                    continue;
                }

                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvbPlaneOutlier[i] = false;
                    continue;
                }

                cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                MapPlane *pNewMP = new MapPlane(p3D, pKF, mpMap);
                pNewMP->AddObservation(pKF,i);
                pKF->AddMapPlane(pNewMP, i);
                pNewMP->UpdateCoefficientsAndPoints();
                mpMap->AddMapPlane(pNewMP);
            }

            mpPointCloudMapping->print();

            cout << "New map created with " << mpMap->MapPlanesInMap() << " planes" << endl;
        }

        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
    }

    void Tracking::SearchLocalPoints() {
// Do not search map points already matched
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPoint *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

// Project points in frame and check its visibility
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
                pMP->IncreaseVisible();
                nToMatch++; //将要match的
            }
        }

        if (nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = 1;
            if (mSensor == System::RGBD)
                th = 3;
            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    void Tracking::SearchLocalLines() {
        // step1：
        for (vector<MapLine *>::iterator vit = mCurrentFrame.mvpMapLines.begin(), vend = mCurrentFrame.mvpMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;
            if (pML) {
                if (pML->isBad()) {
                    *vit = static_cast<MapLine *>(NULL);
                } else {
                    //
                    pML->IncreaseVisible();
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    pML->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        // step2：
        for (vector<MapLine *>::iterator vit = mvpLocalMapLines.begin(), vend = mvpLocalMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;

            if (pML->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pML->isBad())
                continue;

            // step2.1
            if (mCurrentFrame.isInFrustum(pML, 0.6)) {
                pML->IncreaseVisible();
                nToMatch++;
            }
        }

        if (nToMatch > 0) {
            LSDmatcher matcher;
            int th = 1;

            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, th);
        }
    }

    void Tracking::SearchLocalPlanes() {
        for (vector<MapPlane *>::iterator vit = mCurrentFrame.mvpMapPlanes.begin(), vend = mCurrentFrame.mvpMapPlanes.end();
             vit != vend; vit++) {
            MapPlane *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPlane *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                }
            }
        }
    }


    void Tracking::UpdateLocalMap() {
// This is for visualization
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

// Update
        UpdateLocalKeyFrames();
        //cout << "the size of local keyframe" << mvpLocalKeyFrames.size() << endl;

        UpdateLocalPoints();
        UpdateLocalLines();
    }

    void Tracking::UpdateLocalLines() {
        //cout << "Tracking: UpdateLocalLines()" << endl;
        // step1：
        mvpLocalMapLines.clear();

        // step2：
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapLine *> vpMLs = pKF->GetMapLineMatches();

            //step3：将局部关键帧的MapLines添加到mvpLocalMapLines
            for (vector<MapLine *>::const_iterator itML = vpMLs.begin(), itEndML = vpMLs.end();
                 itML != itEndML; itML++) {
                MapLine *pML = *itML;
                if (!pML)
                    continue;
                if (pML->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pML->isBad()) {
                    mvpLocalMapLines.push_back(pML);
                    pML->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    void Tracking::UpdateLocalPoints() {
        mvpLocalMapPoints.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
                 itMP != itEndMP; itMP++) {
                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad()) {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }


    void Tracking::UpdateLocalKeyFrames() {
// Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP->isBad()) {
                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
                         it != itend; it++)
                        keyframeCounter[it->first]++;
                } else {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++) {
            KeyFrame *pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


// Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame *pKF = *itKF;

            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++) {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad()) {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad()) {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame *pParent = pKF->GetParent();
            if (pParent) {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }

        }

        if (pKFmax) {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

/**
 * @details 重定位过程
 * @return true
 * @return false
 *
 * Step 1：计算当前帧特征点的词袋向量
 * Step 2：找到与当前帧相似的候选关键帧
 * Step 3：通过BoW进行匹配
 * Step 4：通过EPnP算法估计姿态
 * Step 5：通过PoseOptimization对姿态进行优化求解
 * Step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
 */
    bool Tracking::Relocalization() {
char oneStep;
oneStep = getchar();
        cout << "Tracking:localization" << endl;
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

        cout << "Tracking,vpCandidateKFs" << vpCandidateKFs.size() << endl;
        if (vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);
        vector<PnPsolver *> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);
        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);
        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;
        for (int i = 0; i < nKFs; i++) {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15) {
                    vbDiscarded[i] = true;
                    continue;
                } else {
                    PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

// Alternatively perform some iterations of P4P RANSAC
// Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch) {
            for (int i = 0; i < nKFs; i++) {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                PnPsolver *pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (!Tcw.empty()) {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++) {
                        if (vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50) {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10,
                                                                      100);

                        if (nadditional + nGood >= 50) {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50) {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                                                                          64);

                                // Final optimization
                                if (nGood + nadditional >= 50) {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50) {
                        bMatch = true;
                        break;
                    }
                }
            }

            if (!bMatch) {

            }
        }

        if (!bMatch) {
            return false;
        } else {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            return true;
        }

    }

    void Tracking::Reset() {
        mpViewer->RequestStop();

        cout << "System Reseting" << endl;
        while (!mpViewer->isStopped())
            usleep(3000);

// Reset Local Mapping
        cout << "Reseting Local Mapper...";
        mpLocalMapper->RequestReset();
        cout << " done" << endl;

// Reset Loop Closing
        cout << "Reseting Loop Closing...";
        mpLoopClosing->RequestReset();
        cout << " done" << endl;

// Clear BoW Database
        cout << "Reseting Database...";
        mpKeyFrameDB->clear();
        cout << " done" << endl;

// Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        if (mpInitializer) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        mpViewer->Release();
    }

    void Tracking::ChangeCalibration(const string &strSettingPath) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool &flag) {
        mbOnlyTracking = flag;
    }

    void Tracking::SaveMesh(const string &filename){
        //
        mpPointCloudMapping->SaveMeshModel(filename);

    }


} //namespace Planar_SLAM
