/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/Planar_SLAM>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"
#include "Common.h"


namespace Planar_SLAM
{
    const int WarpHalfPatchSize = 4;
    const int WarpPatchSize = 8;
class ORBmatcher
{    
public:

    ORBmatcher(float nnratio=0.6, bool checkOri=true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);
    int MatchORBPoints(Frame & CurrentFrame, const Frame &LastFrame);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
     int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12,
                               std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);

    /************************************/
    // 直接法的匹配
    // 用直接法判断能否从在当前图像上找到某地图点的投影
    // 这个函数经常会有误拒的情况，需要进一步检查。
    bool FindDirectProjection(KeyFrame *ref, Frame *curr, MapPoint *mp, Eigen::Vector2f &px_curr, int &search_level);

private:
    // 计算affine wrap矩阵
    void GetWarpAffineMatrix(
            KeyFrame *ref,
            Frame *curr,
            const Vector2f &px_ref,
            MapPoint *mp,
            int level,
            const SE3f &TCR,
            Eigen::Matrix2f &ACR
    );

    inline int GetBestSearchLevel(
            const Eigen::Matrix2f &ACR,
            const int &max_level,
            const KeyFrame *ref
    ) {
        int search_level = 0;
        float D = ACR.determinant();
        while (D > 3.0 && search_level < max_level) {
            search_level += 1;
            D *= ref->mvInvLevelSigma2[1];
        }
        return search_level;
    }

    bool Align2D(
            const cv::Mat &cur_img,
            uint8_t *ref_patch_with_border,
            uint8_t *ref_patch,
            const int n_iter,
            Vector2f &cur_px_estimate);

    void WarpAffine(
            const Eigen::Matrix2f &ACR,
            const cv::Mat &img_ref,
            const Vector2f &px_ref,
            const int &level_ref,
            const KeyFrame *ref,
            const int &search_level,
            const int &half_patch_size,
            uint8_t *patch
    );

    inline uchar GetBilateralInterpUchar(
            const double &x, const double &y, const Mat &gray) {
        const double xx = x - floor(x);
        const double yy = y - floor(y);
        uchar *data = &gray.data[int(y) * gray.step + int(x)];
        return uchar(
                (1 - xx) * (1 - yy) * data[0] +
                xx * (1 - yy) * data[1] +
                (1 - xx) * yy * data[gray.step] +
                xx * yy * data[gray.step + 1]
        );
    }

    uchar _patch[WarpPatchSize * WarpPatchSize];
    // 带边界的，左右各1个像素
    uchar _patch_with_border[(WarpPatchSize + 2) * (WarpPatchSize + 2)];

public:

    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;


protected:

    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);

    float RadiusByViewingCos(const float &viewCos);

    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;//最优评分和次优评分的比例
    bool mbCheckOrientation;//是否检查特征点的方向
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
