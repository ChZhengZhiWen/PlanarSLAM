#include "FrameDrawer.h"
#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<mutex>
#include <pcl/common/distances.h>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM
{

    FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
    {
        mState=Tracking::SYSTEM_NOT_READY;
        mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    }

    cv::Mat FrameDrawer::DrawFrame()
    {
        cv::Mat im;
        vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
        vector<int> vMatches; // Initialization: correspondeces with reference keypoints
        vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
        vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
        int state; // Tracking state

        vector<KeyLine> vCurrentKeyLines;
        vector<KeyLine> vIniKeyLines;
        vector<bool> vbLineVO, vbLineMap;

        //Copy variables within scoped mutex
        {
            unique_lock<mutex> lock(mMutex);
            state=mState;
            if(mState==Tracking::SYSTEM_NOT_READY)
                mState=Tracking::NO_IMAGES_YET;

            mIm.copyTo(im);

            // points and lines for the initialized situation
            if(mState==Tracking::NOT_INITIALIZED)
            {

                vCurrentKeys = mvCurrentKeys;
                vIniKeys = mvIniKeys;
                vMatches = mvIniMatches;
                vCurrentKeyLines = mvCurrentKeyLines;
                vIniKeyLines = mvIniKeyLines;
            }
            // points and lines for the tracking situation
            else if(mState==Tracking::OK)
            {
                vCurrentKeys = mvCurrentKeys;
                vbVO = mvbVO;
                vbMap = mvbMap;
                vCurrentKeyLines = mvCurrentKeyLines;
                vbLineVO = mvbLineVO;
                vbLineMap = mvbLineMap;
            }
            else if(mState==Tracking::LOST)
            {
                vCurrentKeys = mvCurrentKeys;
                vCurrentKeyLines = mvCurrentKeyLines;
            }
        } // destroy scoped mutex -> release mutex

        if(im.channels()<3) //this should be always true
            cvtColor(im,im,CV_GRAY2BGR);

        //Draw
        if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
        {
            for(unsigned int i=0; i<vMatches.size(); i++)
            {
                if(vMatches[i]>=0)
                {
                    cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,cv::Scalar(0,255,0));
                }
            }
        }
        else if(state==Tracking::OK) //TRACKING
        {
            mnTracked=0;
            mnTrackedVO=0;
            const float r = 5;
            const int n = vCurrentKeys.size();
            const int nL = vCurrentKeyLines.size();


            if(1) // visualize 2D points and lines
            {
                //visualize points
                for(int j=0;j<NSNx;j+=1)
                {
                    int u=mvSurfaceNormalx[j].x;
                    int v=mvSurfaceNormalx[j].y;
                    if(u>0&&v>0) {
                        cv::circle(im, cv::Point2f(u, v), 1, cv::Scalar(0, 0, 100), -1);
                    }
                }
                for(int j=0;j<NSNy;j+=1)
                {
                    int u=mvSurfaceNormaly[j].x;
                    int v=mvSurfaceNormaly[j].y;
                    if(u>0&&v>0) {
                        cv::circle(im,cv::Point2f(u,v),1,cv::Scalar(0,100,0),-1);
                    }
                }
                for(int j=0;j<NSNz;j+=1)
                {
                    int u=mvSurfaceNormalz[j].x;
                    int v=mvSurfaceNormalz[j].y;
                    if(u>0&&v>0) {
                        cv::circle(im,cv::Point2f(u,v),1,cv::Scalar(100,0,0),-1);
                    }
                }

                //visualize segmented Manhattan Lines
                // Three colors for three directions
                for(size_t j=0;j<NSLx;j++)
                {
                    int u1 = mvStructLinex[j][2].x; int v1 = mvStructLinex[j][2].y;
                    int u2 = mvStructLinex[j][3].x; int v2 = mvStructLinex[j][3].y;
                    cv::line(im, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(255, 0, 255),4);
                }
                for(size_t j=0;j<NSLy;j++)
                {
                    int u1 = mvStructLiney[j][2].x; int v1 = mvStructLiney[j][2].y;
                    int u2 = mvStructLiney[j][3].x; int v2 = mvStructLiney[j][3].y;
                    cv::line(im, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(0, 255, 0),4);
                }
                for(size_t j=0;j<NSLz;j++)
                {

                    int u1 = mvStructLinez[j][2].x; int v1 = mvStructLinez[j][2].y;
                    int u2 = mvStructLinez[j][3].x; int v2 = mvStructLinez[j][3].y;
                    cv::line(im, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(255, 0, 0),4);
                }


/*
                if (!_IntersectionLine.empty()){
                    for (auto it = _IntersectionLine.begin(); it !=_IntersectionLine.end(); ++it) {
                        if (it->first.first < 0 || it->first.first > curFrame.mvPlaneCoefficients.size() || it->first.second < 0 || it->first.second > curFrame.mvPlaneCoefficients.size())
                            continue;
                        float a = it->second.first(0,0);
                        float b = it->second.first(1,0);
                        float c = it->second.first(2,0);
                        float x0 = it->second.second(0,0);
                        float y0 = it->second.second(1,0);
                        float z0 = it->second.second(2,0);

                        auto planeIndex = it->first;

                        std::vector<PointT> cloudPointsMoreP1;
                        std::vector<PointT> cloudPointsLessP1;
                        std::vector<PointT> cloudPointsMoreP2;
                        std::vector<PointT> cloudPointsLessP2;
                        std::vector<PointT> cloudPointsEdge;

                        for(auto cloudPoints : *curFrame.mvPlanePointsAll[planeIndex.first]){
                            float x = cloudPoints.x;
                            float y = cloudPoints.y;
                            float z = cloudPoints.z;
                            float t = (x - x0) / a;
                            float judgment1 = fabs(t - ((y - y0) / b));
                            float judgment2 = fabs(t - ((z - z0) / c));

                            if ( judgment1 < 0.2 && judgment2 < 0.2 ){
                                cloudPointsMoreP1.push_back(cloudPoints);
                                if ( judgment1 < 0.03 && judgment2 < 0.03 )
                                    cloudPointsLessP1.push_back(cloudPoints);
                            }
                        }
                        for(auto cloudPoints : *curFrame.mvPlanePointsAll[planeIndex.second]){
                            float x = cloudPoints.x;
                            float y = cloudPoints.y;
                            float z = cloudPoints.z;
                            float t = (x - x0) / a;
                            float judgment1 = fabs(t - ((y - y0) / b));
                            float judgment2 = fabs(t - ((z - z0) / c));

                            if ( judgment1 < 0.2 && judgment2 < 0.2 ){
                                cloudPointsMoreP2.push_back(cloudPoints);
                                if ( judgment1 < 0.03 && judgment2 < 0.03 )
                                    cloudPointsLessP2.push_back(cloudPoints);
                            }
                        }

                        float min = 100;
                        for (auto point:cloudPointsLessP1) {
                            for (auto anotherPlanePoint:cloudPointsMoreP2) {
                                float distance = pcl::euclideanDistance(point, anotherPlanePoint);
                                if (distance < min)
                                    min = distance;
                            }
                            if (min < 0.005)
                                cloudPointsEdge.push_back(point);
                        }
                        min = 100;
                        for (auto point:cloudPointsLessP2) {
                            for (auto anotherPlanePoint:cloudPointsMoreP1) {
                                float distance = pcl::euclideanDistance(point, anotherPlanePoint);
                                if (distance < min)
                                    min = distance;
                            }
                            if (min < 0.005)
                                cloudPointsEdge.push_back(point);
                        }

                        for (auto p:cloudPointsEdge) {
                            Vector3f worldPos;
                            worldPos[0]=p.x;
                            worldPos[1]=p.y;
                            worldPos[2]=p.z;
                            Vector2f pixelPos = curFrame.Camera2Pixel(worldPos);
                            float u=pixelPos[0];
                            float v=pixelPos[1];
                            if (u>im.cols || v>im.rows || u <=0.0 ||v<=0.0){
                                continue;
                            }

                            cv::circle(im, cv::Point2f(u, v), 1, cv::Scalar(0, 0, 255), -1);
                        }



                    }
                }
*/

                if (mHavePlaneEdge){
                    for (auto pair:mvAllPlaneEdgeLine) {
                        Vector3f startPoint;
                        Vector3f endPoint;
                        startPoint[0]=pair.first.x;
                        startPoint[1]=pair.first.y;
                        startPoint[2]=pair.first.z;
                        endPoint[0]=pair.second.x;
                        endPoint[1]=pair.second.y;
                        endPoint[2]=pair.second.z;
                        Vector2f pixelStartPoint = curFrame.Camera2Pixel(startPoint);
                        Vector2f pixelEndPoint   = curFrame.Camera2Pixel(endPoint);
                        float su=pixelStartPoint[0];
                        float sv=pixelStartPoint[1];
                        if (su>im.cols || sv>im.rows || su <=0.0 ||sv<=0.0){
                            continue;
                        }
                        float eu=pixelEndPoint[0];
                        float ev=pixelEndPoint[1];
                        if (eu>im.cols || ev>im.rows || eu <=0.0 ||ev<=0.0){
                            continue;
                        }
                        cv::line(im, cv::Point2f(su,sv),cv::Point2f(eu,ev), cv::Scalar(0, 0, 255),4);
                    }
                }



//                for (std::size_t i = 0; i < boundaryIndices.size(); ++i) {
//                    float pointIndex = boundaryIndices[i];
//                    PointT point = _inputCloud->at(pointIndex);
//                    float x = point.x;
//                    float y = point.y;
//                    float z = point.z;
//                    Vector3f worldPos;
//                    worldPos[0]=x;
//                    worldPos[1]=y;
//                    worldPos[2]=z;
//                    Vector2f pixelPos = curFrame.Camera2Pixel(worldPos);
//                    float u=pixelPos[0];
//                    float v=pixelPos[1];
//                    if (u>im.cols || v>im.rows || u <=0.0 ||v<=0.0){
//                        continue;
//                    }
//                    cv::circle(im, cv::Point2f(u, v), 1, cv::Scalar(0, 255, 0), -1);
//                }

            }

            for(int i=0;i<n;i++)
            {
                if(vbVO[i] || vbMap[i])
                {
                    cv::Point2f pt1,pt2;
                    pt1.x=vCurrentKeys[i].pt.x-r;
                    pt1.y=vCurrentKeys[i].pt.y-r;
                    pt2.x=vCurrentKeys[i].pt.x+r;
                    pt2.y=vCurrentKeys[i].pt.y+r;

                    // This is a match to a MapPoint in the map
                    if(vbMap[i])
                    {
                        //cv::rectangle(im,pt1,pt2,cv::Scalar(155,255,155));
                        cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,255,0),-1);
                        mnTracked++;
                    }
                    else // This is match to a "visual odometry" MapPoint created in the last frame
                    {
                        //cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                        cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(155,255,155),-1);
                        mnTrackedVO++;
                    }
                }
            }

            for (int i = 0; i < nL; ++i) {
                if (vbLineVO[i] || vbLineMap[i]) {
                    if (vbLineMap[i])
                        cv::line(im, vCurrentKeyLines[i].getEndPoint(), vCurrentKeyLines[i].getStartPoint(),
                                 cv::Scalar(255, 0, 255),2);
                }
            }


        }
        cv::Mat imWithInfo;
        DrawTextInfo(im,state, imWithInfo);

        return imWithInfo;
    }


    void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
    {
        stringstream s;
        if(nState==Tracking::NO_IMAGES_YET)
            s << " WAITING FOR IMAGES";
        else if(nState==Tracking::NOT_INITIALIZED)
            s << " TRYING TO INITIALIZE ";
        else if(nState==Tracking::OK)
        {
            if(!mbOnlyTracking)
                s << "SLAM MODE |  ";
            else
                s << "LOCALIZATION | ";
            int nKFs = mpMap->KeyFramesInMap();
            int nMPs = mpMap->MapPointsInMap();
            s << "KFs: " << nKFs << ", MPs: " << nMPs << ", mnTracked: " << mnTracked<<", KeyPoints: "<<N;
            if(mnTrackedVO>0)
                s << ", + VO matches: " << mnTrackedVO;
        }
        else if(nState==Tracking::LOST)
        {
            s << " TRACK LOST. TRYING TO RELOCALIZE ";
        }
        else if(nState==Tracking::SYSTEM_NOT_READY)
        {
            s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
        }

        int baseline=0;
        cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

        imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
        im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
        imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
        cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

    }

    void FrameDrawer::Update(Tracking *pTracker)
    {
        unique_lock<mutex> lock(mMutex);
        pTracker->mImGray.copyTo(mIm);
        mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
        N = mvCurrentKeys.size();
        mvbVO = vector<bool>(N,false);
        mvbMap = vector<bool>(N,false);
        mbOnlyTracking = pTracker->mbOnlyTracking;

        mvSurfaceNormalx=pTracker->mCurrentFrame.vSurfaceNormalx;
        mvSurfaceNormaly=pTracker->mCurrentFrame.vSurfaceNormaly;
        mvSurfaceNormalz=pTracker->mCurrentFrame.vSurfaceNormalz;
        NSNx=mvSurfaceNormalx.size();
        NSNy=mvSurfaceNormaly.size();
        NSNz=mvSurfaceNormalz.size();

        mvStructLinex = pTracker->mCurrentFrame.vVanishingLinex;
        mvStructLiney = pTracker->mCurrentFrame.vVanishingLiney;
        mvStructLinez = pTracker->mCurrentFrame.vVanishingLinez;
        NSLx = mvStructLinex.size();
        NSLy = mvStructLiney.size();
        NSLz = mvStructLinez.size();

        mvCurrentKeyLines = pTracker->mCurrentFrame.mvKeylinesUn;
        NL = mvCurrentKeyLines.size();  //自己添加的
        mvbLineVO = vector<bool>(NL, false);
        mvbLineMap = vector<bool>(NL, false);

        mHavePlaneEdge = pTracker->mCurrentFrame.havePlaneEdge;
        mvAllPlaneEdgeLine = pTracker->mCurrentFrame.allPlaneEdgeLine;

        curFrame = pTracker->mCurrentFrame;

        if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
        {
            mvIniKeys=pTracker->mInitialFrame.mvKeys;
            mvIniMatches=pTracker->mvIniMatches;
            //线特征的
            mvIniKeyLines = pTracker->mInitialFrame.mvKeylinesUn;
        }
        else if(pTracker->mLastProcessedState==Tracking::OK)
        {
            for(int i=0;i<N;i++)
            {
                MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                {
                    if(!pTracker->mCurrentFrame.mvbOutlier[i])
                    {
                        if(pMP->Observations()>0)
                            mvbMap[i]=true;
                        else
                            mvbVO[i]=true;
                    }
                }
            }

            for(int i=0; i<NL; i++)
            {
                MapLine* pML = pTracker->mCurrentFrame.mvpMapLines[i];
                if(pML)
                {
                    if(!pTracker->mCurrentFrame.mvbLineOutlier[i])
                    {
                        if(pML->Observations()>0)
                            mvbLineMap[i] = true;
                        else
                            mvbLineVO[i] = true;
                    }
                }
            }


 /* pcl边缘检测
// 计算法向量

            PointCloud::Ptr inputCloud(new PointCloud());
            for (auto plane:curFrame.mvPlanePointsPtr) {
                for (auto point:*plane) {
                    inputCloud->points.push_back(point);
                }
            }

            _inputCloud = inputCloud;

            pcl::NormalEstimation<PointT , pcl::Normal> ne;
            ne.setInputCloud(inputCloud);
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
            ne.setSearchMethod(tree);
            pcl::PointCloud<pcl::Normal>::Ptr input_cloud_normals(new pcl::PointCloud<pcl::Normal>());
            ne.setKSearch(20);  // 设置法线估计的K值
            ne.setViewPoint(0,0,0);
            ne.compute(*input_cloud_normals);

            // 边界点估计
            pcl::BoundaryEstimation<PointT, pcl::Normal, pcl::Boundary> est;
            pcl::PointCloud<pcl::Boundary>::Ptr boundaries(new pcl::PointCloud<pcl::Boundary>());
            est.setInputCloud(inputCloud);
            est.setInputNormals(input_cloud_normals);
            est.setSearchMethod(tree);
//            est.setKSearch(30);
            est.setRadiusSearch(1);  // 设置边界估计的半径
            est.compute(*boundaries);

// 获取边界点索引
            std::vector<int> boundary_indices;
            for (size_t i = 0; i < boundaries->size(); ++i) {
                if ((*boundaries)[i].boundary_point > 0) {
                    boundary_indices.push_back(i);
                }
            }
            boundaryIndices = boundary_indices;

// 使用边界索引提取边界点云数据
            PointCloud::Ptr boundary_cloud(new pcl::PointCloud<PointT>());
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(inputCloud);
            pcl::PointIndices::Ptr boundary_indices_ptr(new pcl::PointIndices());
            boundary_indices_ptr->indices = boundary_indices;
            extract.setIndices(boundary_indices_ptr);
            extract.filter(*boundary_cloud);

 */
        }
        mState=static_cast<int>(pTracker->mLastProcessedState);
    }

} //namespace Planar_SLAM