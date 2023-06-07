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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace Planar_SLAM {

    KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary &voc) :
            mpVoc(&voc) {
        mvInvertedFile.resize(voc.size());
    }

    KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary &voc, const ORBVocabulary &voc_line) :
            mpVoc(&voc), mpVoc_line(&voc_line) {
        mvInvertedFile.resize(voc.size() + voc_line.size());
    }


    void KeyFrameDatabase::add(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutex);

        for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
            mvInvertedFile[vit->first].push_back(pKF);
    }

    void KeyFrameDatabase::add_wh(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutex);

        for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
            mvInvertedFile[vit->first].push_back(pKF);
        for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec_line.begin(), vend = pKF->mBowVec_line.end();
             vit != vend; vit++)
            mvInvertedFile[vit->first].push_back(pKF);
    }

    void KeyFrameDatabase::erase(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutex);

        // Erase elements in the Inverse File for the entry
        for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end();
             vit != vend; vit++) {
            // List of keyframes that share the word
            list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

            for (list<KeyFrame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
                if (pKF == *lit) {
                    lKFs.erase(lit);
                    break;
                }
            }
        }
    }

    void KeyFrameDatabase::clear() {
        mvInvertedFile.clear();
        mvInvertedFile.resize(mpVoc->size());
    }


    vector<KeyFrame *> KeyFrameDatabase::DetectLoopCandidates(KeyFrame *pKF, float minScore) {
        set<KeyFrame *> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
        list<KeyFrame *> lKFsSharingWords;

        // Search all keyframes that share a word with current keyframes
        // Discard keyframes connected to the query keyframe
        {
            unique_lock<mutex> lock(mMutex);

            for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end();
                 vit != vend; vit++) {
                list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

                for (list<KeyFrame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
                    KeyFrame *pKFi = *lit;
                    if (pKFi->mnLoopQuery != pKF->mnId) {
                        pKFi->mnLoopWords = 0;
                        if (!spConnectedKeyFrames.count(pKFi)) {
                            pKFi->mnLoopQuery = pKF->mnId;
                            lKFsSharingWords.push_back(pKFi);
                        }
                    }
                    pKFi->mnLoopWords++;
                }
            }
        }

        if (lKFsSharingWords.empty())
            return vector<KeyFrame *>();

        list<pair<float, KeyFrame *> > lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords = 0;
        for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
             lit != lend; lit++) {
            if ((*lit)->mnLoopWords > maxCommonWords)
                maxCommonWords = (*lit)->mnLoopWords;
        }

        int minCommonWords = maxCommonWords * 0.8f;

        int nscores = 0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
             lit != lend; lit++) {
            KeyFrame *pKFi = *lit;

            if (pKFi->mnLoopWords > minCommonWords) {
                nscores++;

                float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);

                pKFi->mLoopScore = si;
                if (si >= minScore)
                    lScoreAndMatch.push_back(make_pair(si, pKFi));
            }
        }

        if (lScoreAndMatch.empty())
            return vector<KeyFrame *>();

        list<pair<float, KeyFrame *> > lAccScoreAndMatch;
        float bestAccScore = minScore;

        // Lets now accumulate score by covisibility
        for (list<pair<float, KeyFrame *> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end();
             it != itend; it++) {
            KeyFrame *pKFi = it->second;
            vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float bestScore = it->first;
            float accScore = it->first;
            KeyFrame *pBestKF = pKFi;
            for (vector<KeyFrame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
                KeyFrame *pKF2 = *vit;
                if (pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords > minCommonWords) {
                    accScore += pKF2->mLoopScore;
                    if (pKF2->mLoopScore > bestScore) {
                        pBestKF = pKF2;
                        bestScore = pKF2->mLoopScore;
                    }
                }
            }

            lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
            if (accScore > bestAccScore)
                bestAccScore = accScore;
        }

        // Return all those keyframes with a score higher than 0.75*bestScore
        float minScoreToRetain = 0.75f * bestAccScore;

        set<KeyFrame *> spAlreadyAddedKF;
        vector<KeyFrame *> vpLoopCandidates;
        vpLoopCandidates.reserve(lAccScoreAndMatch.size());

        for (list<pair<float, KeyFrame *> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end();
             it != itend; it++) {
            if (it->first > minScoreToRetain) {
                KeyFrame *pKFi = it->second;
                if (!spAlreadyAddedKF.count(pKFi)) {
                    vpLoopCandidates.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }


        return vpLoopCandidates;
    }

    vector<KeyFrame *>
    KeyFrameDatabase::DetectLoopCandidates_zzw(KeyFrame *pKF, float minScore, const vector<KeyFrame *> &allKeyFrame,
                                               vector<float> avgLineManScore) {

//        cout<<"minScore "<<minScore<<endl;
        vector<KeyFrame *> vpLoopCandidates_;
        {
            unique_lock<mutex> lock(mMutex);


//            vector<float> cur_line_manhattan = computeLineManhattan(pKF);
            const DBoW2::BowVector &CurrentBowVec = pKF->mBowVec;
            int len = allKeyFrame.size();
            for (int i = 0; i < len - 1100; ++i) {
//                if (abs(int(pKF->mnId - kfi->mnId)) < 600)
//                    continue;

//                vector<KeyFrame*> vpCandidateKFsManhattan;
//
//                vector<float> lKFs_line_manhattan = computeLineManhattan(allKeyFrame[i]);
//                int num = 0;
//                vector<float> differ = {-99, -99, -99};
//                for (int j = 0; j < 3; ++j) {
//                    if (cur_line_manhattan[j] == -1 || lKFs_line_manhattan[j] == -1)
//                        continue;
//                    differ[j] = abs(cur_line_manhattan[j] - lKFs_line_manhattan[j]);
//                    float _min;
//                    if (avgLineManScore[j] < 0.1) {
//                        _min = avgLineManScore[j];
//                    } else {
//                        _min = 0.1;
//                    }
//                    if (differ[j] <= _min)
//                        num++;
//                }
//
//                if (num < 2)
//                    continue;


                const DBoW2::BowVector &BowVec = allKeyFrame[i]->mBowVec;
                float si = mpVoc->score(CurrentBowVec, BowVec);
                if (si >= minScore) {
                    cout << si << " ";
                    vpLoopCandidates_.push_back(allKeyFrame[i]);
                }
            }
            cout << endl;
        }
        if (vpLoopCandidates_.empty())
            return vector<KeyFrame *>();
        return vpLoopCandidates_;


        set<KeyFrame *> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
        list<KeyFrame *> lKFsSharingWords;

        // Search all keyframes that share a word with current keyframes
        // Discard keyframes connected to the query keyframe
        {
            unique_lock<mutex> lock(mMutex);
            for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end();
                 vit != vend; vit++) {
                list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

                for (list<KeyFrame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
                    KeyFrame *pKFi = *lit;
                    if (pKFi->mnLoopQuery != pKF->mnId) {
                        pKFi->mnLoopWords = 0;
                        if (!spConnectedKeyFrames.count(pKFi)) {
                            pKFi->mnLoopQuery = pKF->mnId;
                            lKFsSharingWords.push_back(pKFi);
                        }
                    }
                    pKFi->mnLoopWords++;
                }
            }
        }

        if (lKFsSharingWords.empty())
            return vector<KeyFrame *>();

        list<pair<float, KeyFrame *> > lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords = 0;

        for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
             lit != lend; lit++) {
            if ((*lit)->mnLoopWords > maxCommonWords)
                maxCommonWords = (*lit)->mnLoopWords;
        }

        int minCommonWords = maxCommonWords * 0.8f;

        int nscores = 0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
             lit != lend; lit++) {
            KeyFrame *pKFi = *lit;

            if (pKFi->mnLoopWords > minCommonWords) {
                nscores++;

                float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);

                ///modify by wh
                float sl = mpVoc_line->score(pKF->mBowVec_line, pKFi->mBowVec_line);
//                si = 1.0 * pKF->N / (pKF->N + pKF->NL) * si + 1.0 * pKF->NL / (pKF->N + pKF->NL) * sl;

                pKFi->mLoopScore = si;
                if (si >= minScore)
                    lScoreAndMatch.push_back(make_pair(si, pKFi));
            }
        }

        if (lScoreAndMatch.empty())
            return vector<KeyFrame *>();

        list<pair<float, KeyFrame *> > lAccScoreAndMatch;
        float bestAccScore = minScore;

        // Lets now accumulate score by covisibility
        for (list<pair<float, KeyFrame *> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end();
             it != itend; it++) {
            KeyFrame *pKFi = it->second;
            vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float bestScore = it->first;
            float accScore = it->first;
            KeyFrame *pBestKF = pKFi;
            for (vector<KeyFrame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
                KeyFrame *pKF2 = *vit;
                if (pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords > minCommonWords) {
                    accScore += pKF2->mLoopScore;
                    if (pKF2->mLoopScore > bestScore) {
                        pBestKF = pKF2;
                        bestScore = pKF2->mLoopScore;
                    }
                }
            }

            lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
            if (accScore > bestAccScore)
                bestAccScore = accScore;
        }

        // Return all those keyframes with a score higher than 0.75*bestScore
        float minScoreToRetain = 0.75f * bestAccScore;

        set<KeyFrame *> spAlreadyAddedKF;
        vector<KeyFrame *> vpLoopCandidates;
        vpLoopCandidates.reserve(lAccScoreAndMatch.size());

        for (list<pair<float, KeyFrame *> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end();
             it != itend; it++) {
            if (it->first > minScoreToRetain) {
                KeyFrame *pKFi = it->second;
                if (!spAlreadyAddedKF.count(pKFi)) {
                    vpLoopCandidates.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }


        return vpLoopCandidates;
    }

    vector<KeyFrame *>
    KeyFrameDatabase::DetectLoopCandidatesAllkeyframe(KeyFrame *pKF, float minScore, vector<KeyFrame *> allKF,vector<float> avgLineManScore) {
        vector<KeyFrame *> ret;
        map<float,KeyFrame *> candidatesRet;
        vector<float> scoreForSort;

        int len = allKF.size();
        vector<float> cur_line_manhattan = pKF->line_manhattan_err;
        for (int i = 0; i < len - 1000; i++) {

            vector<float> lKFs_line_manhattan = allKF[i]->line_manhattan_err;
            int num = 0;
            vector<float> differ = {-99, -99, -99};
            for (int j = 0; j < 3; ++j) {
                if (cur_line_manhattan[j] == -1 || lKFs_line_manhattan[j] == -1)
                    continue;
                differ[j] = abs(cur_line_manhattan[j] - lKFs_line_manhattan[j]);
                float _min;
                if (avgLineManScore[j] < 0.1) {
                    _min = avgLineManScore[j];
                } else {
                    _min = 0.1;
                }
                if (differ[j] <= _min)
                    num++;
            }

            if (num < 2)
                continue;

            float score = mpVoc->score(pKF->mBowVec, allKF[i]->mBowVec);
            cout<<score<<" ";
            if (score > minScore)
                ret.push_back(allKF[i]);
//            if (score > minScore){
//                scoreForSort.push_back(score);
//                candidatesRet[score]=allKF[i];
//            }
        }


//        if (!scoreForSort.empty()){
//            sort(scoreForSort.rbegin(), scoreForSort.rend());
//            for (int i = 0; i < min(50,int(scoreForSort.size())); ++i) {
//                ret.push_back(candidatesRet[scoreForSort[i]]);
//            }
//        }

        return ret;
    }

    vector<KeyFrame *> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F) {
        list<KeyFrame *> lKFsSharingWords;

        // Search all keyframes that share a word with current frame
        {
            unique_lock<mutex> lock(mMutex);

            for (DBoW2::BowVector::const_iterator vit = F->mBowVec.begin(), vend = F->mBowVec.end();
                 vit != vend; vit++) {
                list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

                for (list<KeyFrame *>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
                    KeyFrame *pKFi = *lit;
                    if (pKFi->mnRelocQuery != F->mnId) {
                        pKFi->mnRelocWords = 0;
                        pKFi->mnRelocQuery = F->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                    pKFi->mnRelocWords++;
                }
            }
        }
        if (lKFsSharingWords.empty())
            return vector<KeyFrame *>();

        // Only compare against those keyframes that share enough words
        int maxCommonWords = 0;
        for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
             lit != lend; lit++) {
            if ((*lit)->mnRelocWords > maxCommonWords)
                maxCommonWords = (*lit)->mnRelocWords;
        }

        int minCommonWords = maxCommonWords * 0.8f;

        list<pair<float, KeyFrame *> > lScoreAndMatch;

        int nscores = 0;

        // Compute similarity score.
        for (list<KeyFrame *>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
             lit != lend; lit++) {
            KeyFrame *pKFi = *lit;

            if (pKFi->mnRelocWords > minCommonWords) {
                nscores++;
                float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
                pKFi->mRelocScore = si;
                lScoreAndMatch.push_back(make_pair(si, pKFi));
            }
        }

        if (lScoreAndMatch.empty())
            return vector<KeyFrame *>();

        list<pair<float, KeyFrame *> > lAccScoreAndMatch;
        float bestAccScore = 0;

        // Lets now accumulate score by covisibility
        for (list<pair<float, KeyFrame *> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end();
             it != itend; it++) {
            KeyFrame *pKFi = it->second;
            vector<KeyFrame *> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float bestScore = it->first;
            float accScore = bestScore;
            KeyFrame *pBestKF = pKFi;
            for (vector<KeyFrame *>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
                KeyFrame *pKF2 = *vit;
                if (pKF2->mnRelocQuery != F->mnId)
                    continue;

                accScore += pKF2->mRelocScore;
                if (pKF2->mRelocScore > bestScore) {
                    pBestKF = pKF2;
                    bestScore = pKF2->mRelocScore;
                }

            }
            lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
            if (accScore > bestAccScore)
                bestAccScore = accScore;
        }

        // Return all those keyframes with a score higher than 0.75*bestScore
        float minScoreToRetain = 0.75f * bestAccScore;
        set<KeyFrame *> spAlreadyAddedKF;
        vector<KeyFrame *> vpRelocCandidates;
        vpRelocCandidates.reserve(lAccScoreAndMatch.size());
        for (list<pair<float, KeyFrame *> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end();
             it != itend; it++) {
            const float &si = it->first;
            if (si > minScoreToRetain) {
                KeyFrame *pKFi = it->second;
                if (!spAlreadyAddedKF.count(pKFi)) {
                    vpRelocCandidates.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }

        return vpRelocCandidates;
    }


    vector<float> KeyFrameDatabase::computeLineManhattan(KeyFrame *frame) {
        unique_lock<mutex> lock(mMutex);
        vector<float> res;
        vector<cv::Mat> parallelLine1, parallelLine2, parallelLine3;
        for (int i = 0; i < 3; ++i) {
            cv::Mat manhattanAxis = frame->mvManhattanForLoop[i];

            for (int j = 0; j < frame->mvLines3D.size(); ++j) {
                Vector6d lineVector = frame->obtain3DLine(j);

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
                //单位向量
                mapLine /= cv::norm(mapLine);

                float angle = mapLine.at<float>(0) * manhattanAxis.at<float>(0) +
                              mapLine.at<float>(1) * manhattanAxis.at<float>(1) +
                              mapLine.at<float>(2) * manhattanAxis.at<float>(2);

                //0.86603 degree 30
                //0.96593 degree 15
                float parallelThr = 0.86603;
                if (angle > parallelThr || angle < -parallelThr) {
                    if (i == 0)
                        parallelLine1.push_back(mapLine);
                    else if (i == 1)
                        parallelLine2.push_back(mapLine);
                    else
                        parallelLine3.push_back(mapLine);
                }
            }

            vector<cv::Mat> parallelLine;
            if (i == 0)
                parallelLine = parallelLine1;
            else if (i == 1)
                parallelLine = parallelLine2;
            else
                parallelLine = parallelLine3;

            float sum_chi2 = 0.0;
            if (parallelLine.empty()) {
                res.push_back(-1);
            } else {
                for (auto &it: parallelLine) {
                    float angle = it.at<float>(0) * manhattanAxis.at<float>(0) +
                                  it.at<float>(1) * manhattanAxis.at<float>(1) +
                                  it.at<float>(2) * manhattanAxis.at<float>(2);
                    if (angle < 0)
                        it = -it;

                    cv::Mat R_line, R_manhattan;
                    cv::Rodrigues(it, R_line);
                    cv::Rodrigues(manhattanAxis, R_manhattan);
                    auto _chi2 = abs(acos((cv::trace(R_line.inv() * R_manhattan)[0] - 1.0) / 2.0));
                    sum_chi2 += _chi2;
                }
                res.push_back(sum_chi2 / parallelLine.size());
            }
        }
        return res;
    }
} //namespace Planar_SLAM
