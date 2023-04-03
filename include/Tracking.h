#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"


#include <mutex>

#include "auxiliar.h"
//#include "ExtractLineSegment.h"
#include "MapLine.h"
#include "LSDmatcher.h"
#include "PlaneMatcher.h"

#include "SparseImageAlign.h"

#include "MeshViewer.h"
#include "MapPlane.h"

#include <Thirdparty/sophus/sophus/se3.hpp>
#include <Thirdparty/sophus/sophus/so3.hpp>


class MeshViewer;

namespace Planar_SLAM
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    // TO DO

    
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    //obtain rotation matrix from the Manhattan World assumption
    cv::Mat SeekManhattanFrame(vector<SurfaceNormal>  &vSurfaceNormal,vector<FrameLine>&vVanishingDirection);
    cv::Mat TrackManhattanFrame(cv::Mat &mLastRcm,vector<SurfaceNormal> &vSurfaceNormal,vector<FrameLine>&vVanishingDirection);

    //
    sMS MeanShift(vector<cv::Point2d> & v2D);
    cv::Mat ProjectSN2MF(int a,const cv::Mat &R_cm,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection);
    ResultOfMS ProjectSN2MF(int a,const cv::Mat &R_cm,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection,const int numOfSN);
    axiSNV ProjectSN2Conic(int a,const cv::Mat &R_cm,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection);

    cv::Mat ClusterMultiManhattanFrame(vector<cv::Mat> &vRotationCandidate,double &clusterRatio);
    vector<vector<int>>  EasyHist(vector<float> &vDistance,int &histStart,float &histStep,int&histEnd);
    void SaveMesh(const string &filename);

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // update MF rotation
    bool mUpdateMF;
    // Input sensor
    int mSensor;


    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImRGB;
    cv::Mat mImGray;
    cv::Mat mImDepth;
    cv::Mat mImDepth_CV_32F;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    vector<pair<int, int>> mvLineMatches;
    vector<cv::Point3f> mvLineS3D;   //start point
    vector<cv::Point3f> mvLineE3D;   //end point
    vector<bool> mvbLineTriangulated;   //
    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;


    cv::Mat Rotation_cm;
    cv::Mat mRotation_wc;
    cv::Mat mLastRcm;
    cv::Mat Rotation_gc;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

    shared_ptr<MeshViewer>  mpPointCloudMapping;

    double getTrackTime();
    double trackTime;

    ///-----------------------------------------
    bool fullManhattanFound;
//    float mfMFVerTh = 0.01745;//原本应该放在配置文件中，这里为了方便直接赋值了 89
//    float mfMFVerTh = 0.08716;//改为pSLAM的参数 85
//    float mfMFVerTh = 0.0523;//87
    float mfMFVerTh = 0.0349;//88
    cv::Mat manhattanRcw;

    SparseImgAlign *mpAlign = nullptr;

    bool bManhattan;

    std::mutex mMutexLoopStop;
    bool loopStop;

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();
    void Track_zzw();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();
    void StereoInitialization_zzw();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();
    void CreateInitialMapMonoWithLine();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    bool TranslationEstimation();
    bool TranslationEstimation_MW();
    bool TranslationEstimation_ygz();
    bool TranslationWithMotionModel();
    bool TranslationWithMotionModel_MW();
    bool TranslationWithMotionModel_NMW();
    bool Relocalization();

    bool DetectManhattan();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalLines();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();
    void SearchLocalLines();
    void SearchLocalPlanes();


    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // 用SVO中的sparse alignment 来更新当前帧位姿
    // 但是这里不会处理特征点的关联和投影关系
    bool TrackWithSparseAlignment();

    // 直接法的 Local Map 追踪
    // 与局部地图的直接匹配
    bool TrackLocalMapDirect();

    void SearchLocalPointsDirect();

    // 从地图观测中选取近的几个观测
    /**
     * @param[in] observations 地图点的观测数据
     * @param[in] n 选取的数量
     */
    vector<pair<KeyFrame *, size_t> > SelectNearestKeyframe(const map<KeyFrame *, size_t> &observations, int n = 5);


    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;


    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    std::vector<MapLine*> mvpLocalMapLines;

    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;
    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    //自己添加的，两个用于纠正畸变的映射矩阵
    cv::Mat mUndistX, mUndistY;
    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;
    int mnLineMatchesInliers;   //线特征

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;
    list<MapLine*>mlpTemporalLines;

    float mfDThRef;
    float mfDThMon;
    float mfAThRef;
    float mfAThMon;

    float mfVerTh;
    float mfParTh;

    int manhattanCount;
    int fullManhattanCount;

    set<MapPoint *> mvpDirectMapPointsCache;     // 缓存之前匹配到的地图点
    bool mbDirectFailed = false;    // 直接方法是否失败了？

    int failedNum ;

};

} //namespace ORB_SLAM

#endif // TRACKING_H
