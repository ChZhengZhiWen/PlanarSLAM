#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"
#include "Frame.h"
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include<mutex>


namespace Planar_SLAM
{

    class Tracking;
    class Viewer;

    class FrameDrawer
    {
    public:
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud <PointT> PointCloud;

        FrameDrawer(Map* pMap);

        // Update info from the last processed frame.
        void Update(Tracking *pTracker);

        // Draw last processed frame.
        cv::Mat DrawFrame();



        std::vector<int> boundaryIndices;
        PointCloud::Ptr _inputCloud;

        std::vector< pair<PointT,PointT> > mvAllPlaneEdgeLine;
        bool mHavePlaneEdge;

    protected:

        void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

        // Info of the frame to be drawn
        cv::Mat mIm;
        int N;
        std::vector<cv::KeyPoint> mvCurrentKeys;
        std::vector<bool> mvbMap, mvbVO;
        bool mbOnlyTracking;
        int mnTracked, mnTrackedVO;
        std::vector<cv::KeyPoint> mvIniKeys;
        std::vector<int> mvIniMatches;
        int mState;

        int NL;
        std::vector<cv::line_descriptor::KeyLine> mvCurrentKeyLines;
        std::vector<bool> mvbLineMap, mvbLineVO;
        std::vector<cv::line_descriptor::KeyLine> mvIniKeyLines;

        int NSNx;int NSNz;int NSNy;

        std::vector<cv::Point2i>mvSurfaceNormalx;std::vector<cv::Point2i>mvSurfaceNormaly;std::vector<cv::Point2i>mvSurfaceNormalz;

        int NSLx;int NSLz;int NSLy;

        std::vector<vector<cv::Point2d>>mvStructLinex;std::vector<vector<cv::Point2d>>mvStructLiney;std::vector<vector<cv::Point2d>>mvStructLinez;
        vector<cv::line_descriptor::KeyLine> vCurrentKeyLines;

        Map* mpMap;

        std::mutex mMutex;

        Frame curFrame;
    };

} //namespace Planar_SLAM

#endif // FRAMEDRAWER_H