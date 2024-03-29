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

#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>

#include "KeyFrame.h"
#include "Frame.h"
#include "ORBVocabulary.h"

#include<mutex>


namespace Planar_SLAM
{

class KeyFrame;
class Frame;


class KeyFrameDatabase
{
public:

    KeyFrameDatabase(const ORBVocabulary &voc);
    KeyFrameDatabase(const ORBVocabulary &voc,const ORBVocabulary &voc_line);

   void add(KeyFrame* pKF);

    void add_wh(KeyFrame* pKF);

   void erase(KeyFrame* pKF);

   void clear();

   // Loop Detection
   std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);
   std::vector<KeyFrame *> DetectLoopCandidatesAllkeyframe(KeyFrame* pKF, float minScore,vector<KeyFrame*> vec,vector<float> avgLineManScore);
   std::vector<KeyFrame *> DetectLoopCandidates_zzw(KeyFrame* pKF, float minScore,const vector<KeyFrame*>& allKeyFrame, vector<float> avgLineManScore);

   // Relocalization
   std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);

   vector<float> computeLineManhattan(KeyFrame *frame);

protected:

  // Associated vocabulary
  const ORBVocabulary* mpVoc;

  const ORBVocabulary* mpVoc_line;

  // Inverted file
  std::vector<list<KeyFrame*> > mvInvertedFile;

  // Mutex
  std::mutex mMutex;
};

} //namespace Planar_SLAM

#endif
