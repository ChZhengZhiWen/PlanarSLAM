#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM
{


    MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
        mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
        mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
        mPointSize = fSettings["Viewer.PointSize"];
        mCameraSize = fSettings["Viewer.CameraSize"];
        mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
        mLineWidth = fSettings["Viewer.LineWidth"];
    }

    void MapDrawer::DrawMapPoints()
    {
        const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
        const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

        set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

        if(vpMPs.empty())
            return;

        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,0.0);

        for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
        {
            if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
                continue;
            cv::Mat pos = vpMPs[i]->GetWorldPos();
            glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
        }
        glEnd();

        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,0.0);     //红色

        for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
        {
            if((*sit)->isBad())
                continue;
            cv::Mat pos = (*sit)->GetWorldPos();
            glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
        }

        glEnd();
    }

    void MapDrawer::DrawMapLines()
    {
        const vector<MapLine*> &vpMLs = mpMap->GetAllMapLines();
        const vector<MapLine*> &vpRefMLs = mpMap->GetReferenceMapLines();

        set<MapLine*> spRefMLs(vpRefMLs.begin(), vpRefMLs.end());

        if(vpMLs.empty())
            return;

//        glLineWidth(mLineWidth);
//        glBegin ( GL_LINES );
//        glColor3f(0.0,0.0,0.0);
//
//        for(size_t i=0, iend=vpMLs.size(); i<iend; i++)
//        {
//            if(vpMLs[i]->isBad() || spRefMLs.count(vpMLs[i]))
//                continue;
//            Vector6d pos = vpMLs[i]->GetWorldPos();
//            glVertex3f(pos(0), pos(1), pos(2));
//            glVertex3f(pos(3), pos(4), pos(5));
//
//        }
//        glEnd();

//        glLineWidth(mLineWidth);
//        glBegin ( GL_LINES );
//        glColor3f(255,0.0,0.0); //红色
//cout<<"*********************************"<<endl;
//        for(set<MapLine*>::iterator sit=spRefMLs.begin(), send=spRefMLs.end(); sit!=send; sit++)
//        {
//            if((*sit)->isBad())
//                continue;
//            Vector6d pos = (*sit)->GetWorldPos();
//            glVertex3f(pos(0), pos(1), pos(2));
//            glVertex3f(pos(3), pos(4), pos(5));
//            cout<<pos<<endl<<endl;
//        }
//        glEnd();
//cout<<"*********************************"<<endl;

        const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        for(size_t i=0; i<vpKFs.size(); i++){
            KeyFrame* pKF = vpKFs[i];
            if (i == vpKFs.size()-1){

                auto tem = pKF->mvManhattanForLoop;
                glLineWidth(1);
                glBegin ( GL_LINES );
                glColor3f(255,0.0,0.0); //红色
                glVertex3f(0.0, 0.0, 0.0);
                glVertex3f(tem[0].at<float>(0) ,tem[0].at<float>(1) ,tem[0].at<float>(2));
                glEnd();
                glLineWidth(1);
                glBegin ( GL_LINES );
                glColor3f(0.0,255,0.0); //红色
                glVertex3f(0.0, 0.0, 0.0);
                glVertex3f(tem[1].at<float>(0) ,tem[1].at<float>(1) ,tem[1].at<float>(2));
                glEnd();
                glLineWidth(1);
                glBegin ( GL_LINES );
                glColor3f(0.0,0.0,255); //红色
                glVertex3f(0.0, 0.0, 0.0);
                glVertex3f(tem[2].at<float>(0) ,tem[2].at<float>(1) ,tem[2].at<float>(2));
                glEnd();

                for (int j = 0; j < pKF->mvLines3D.size(); ++j) {
                    Vector6d lineVector = pKF->obtain3DLine(j);

                    cv::Mat startPoint = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                    cv::Mat endPoint = cv::Mat::eye(cv::Size(1, 3), CV_32F);

                    startPoint.at<float>(0, 0) = lineVector[0];
                    startPoint.at<float>(1, 0) = lineVector[1];
                    startPoint.at<float>(2, 0) = lineVector[2];
                    endPoint.at<float>(0, 0) = lineVector[3];
                    endPoint.at<float>(1, 0) = lineVector[4];
                    endPoint.at<float>(2, 0) = lineVector[5];

                    cv::Mat mapLine = startPoint - endPoint;

                    if (mapLine.at<float>(0) == 0 && mapLine.at<float>(1) == 0 && mapLine.at<float>(2) == 0)
                        continue;

                    mapLine /= cv::norm(mapLine);

                    float angle1 = mapLine.at<float>(0) * tem[0].at<float>(0) +
                                  mapLine.at<float>(1) * tem[0].at<float>(1) +
                                  mapLine.at<float>(2) * tem[0].at<float>(2);
                    float angle2 = mapLine.at<float>(0) * tem[1].at<float>(0) +
                                  mapLine.at<float>(1) * tem[1].at<float>(1) +
                                  mapLine.at<float>(2) * tem[1].at<float>(2);
                    float angle3 = mapLine.at<float>(0) * tem[2].at<float>(0) +
                                  mapLine.at<float>(1) * tem[2].at<float>(1) +
                                  mapLine.at<float>(2) * tem[2].at<float>(2);
                    float parallelThr = 0.86603 ;
                    if (angle1 > parallelThr || angle1 < - parallelThr){
                        glLineWidth(1);
                        glBegin ( GL_LINES );
                        glColor3f(255,0.0,0.0);
                        glVertex3f(lineVector[0],lineVector[1],lineVector[2]);
                        glVertex3f(lineVector[3],lineVector[4],lineVector[5]);
                        glEnd();
                    }
                    else if (angle2 > parallelThr || angle2 < - parallelThr){
                        glLineWidth(1);
                        glBegin ( GL_LINES );
                        glColor3f(0.0,255,0.0);
                        glVertex3f(lineVector[0],lineVector[1],lineVector[2]);
                        glVertex3f(lineVector[3],lineVector[4],lineVector[5]);
                        glEnd();
                    }
                    else if (angle3 > parallelThr || angle3 < - parallelThr){
                        glLineWidth(1);
                        glBegin ( GL_LINES );
                        glColor3f(0.0,0.0,255);
                        glVertex3f(lineVector[0],lineVector[1],lineVector[2]);
                        glVertex3f(lineVector[3],lineVector[4],lineVector[5]);
                        glEnd();
                    }




                }
            }
        }

//        glLineWidth(1);
//        glBegin ( GL_LINES );
//        glColor3f(255,0.0,0.0); //红色
//
//        glVertex3f(0.0, 0.0, 0.0);
//        glVertex3f(0.987295 ,-0.0417315 ,0.152679);
//
//        glVertex3f(0.0, 0.0, 0.0);
//        glVertex3f(-0.0292084 ,0.900017 ,0.434876);
//
//        glVertex3f(0.0, 0.0, 0.0);
//        glVertex3f(-0.155971 ,-0.421205 ,0.893454);
//        glEnd();
//
//        glLineWidth(1);
//        glBegin ( GL_LINES );
//        glColor3f(0.0,255,0.0); //红色
//
//        glVertex3f(0.0, 0.0, 0.0);
//        glVertex3f(0.99661 ,0.041096 ,-0.0712768);
//
//        glVertex3f(0.0, 0.0, 0.0);
//        glVertex3f(-0.00182036 ,0.869643 ,0.493678);
//
//        glVertex3f(0.0, 0.0, 0.0);
//        glVertex3f(0.0822735 ,-0.491874 ,0.866769);
//        glEnd();
    }

    void MapDrawer::DrawMapPlanes() {
        const vector<MapPlane*> &vpMPs = mpMap->GetAllMapPlanes();
        if(vpMPs.empty())
            return;
        glPointSize(mPointSize*2);
        //glBegin(GL_POINTS);
        //glBegin(GL_TRIANGLES);
        for(auto pMP : vpMPs){
            float ir = pMP->mRed;
            float ig = pMP->mGreen;
            float ib = pMP->mBlue;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            glColor3f(ir/norm, ig/norm, ib/norm);

            glBegin(GL_POINTS);
            cv::Mat pos(4,1,CV_32F);
            for(auto& p : pMP->mvPlanePoints.get()->points){
                pos.at<float>(0) = p.x;
                pos.at<float>(1) = p.y;
                pos.at<float>(2) = p.z;
                pos.at<float>(3) = 1;

                glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
            }
            glDisable(GL_LIGHTING);
            glEnd();
        }

    }

    void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
    {
        const float &w = mKeyFrameSize;
        const float h = w*0.75;
        const float z = w*0.6;

        const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

        if(bDrawKF)
        {
            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                cv::Mat Twc = pKF->GetPoseInverse().t();

                glPushMatrix();

                glMultMatrixf(Twc.ptr<GLfloat>(0));

                glLineWidth(mKeyFrameLineWidth);
                glColor3f(0.0f,0.0f,1.0f);
                glBegin(GL_LINES);
                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }

        if(bDrawGraph)
        {
            glLineWidth(mGraphLineWidth);
            glColor4f(0.0f,1.0f,0.0f,0.6f);
            glBegin(GL_LINES);

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                // Covisibility Graph
                const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
                cv::Mat Ow = vpKFs[i]->GetCameraCenter();
                if(!vCovKFs.empty())
                {
                    for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                    {
                        if((*vit)->mnId<vpKFs[i]->mnId)
                            continue;
                        cv::Mat Ow2 = (*vit)->GetCameraCenter();
                        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                        glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                    }
                }

                // Spanning tree
                KeyFrame* pParent = vpKFs[i]->GetParent();
                if(pParent)
                {
                    cv::Mat Owp = pParent->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
                }

                // Loops
                set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
                for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
                {
                    if((*sit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Owl = (*sit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
                }
            }

            glEnd();
        }
    }

    void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
    {
        const float &w = mCameraSize;
        const float h = w*0.75;
        const float z = w*0.6;

        glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f,1.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }


    void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
    {
        unique_lock<mutex> lock(mMutexCamera);
        mCameraPose = Tcw.clone();
    }

    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
    {
        if(!mCameraPose.empty())
        {
            cv::Mat Rwc(3,3,CV_32F);
            cv::Mat twc(3,1,CV_32F);
            {
                unique_lock<mutex> lock(mMutexCamera);
                Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
                twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
            }

            M.m[0] = Rwc.at<float>(0,0);
            M.m[1] = Rwc.at<float>(1,0);
            M.m[2] = Rwc.at<float>(2,0);
            M.m[3]  = 0.0;

            M.m[4] = Rwc.at<float>(0,1);
            M.m[5] = Rwc.at<float>(1,1);
            M.m[6] = Rwc.at<float>(2,1);
            M.m[7]  = 0.0;

            M.m[8] = Rwc.at<float>(0,2);
            M.m[9] = Rwc.at<float>(1,2);
            M.m[10] = Rwc.at<float>(2,2);
            M.m[11]  = 0.0;

            M.m[12] = twc.at<float>(0);
            M.m[13] = twc.at<float>(1);
            M.m[14] = twc.at<float>(2);
            M.m[15]  = 1.0;
        }
        else
            M.SetIdentity();
    }

} //namespace Planar_SLAM