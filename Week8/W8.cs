﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LearningFromData;
using libsvm;

namespace Week8
{
    class W8
    {
        static void Main( string[] args )
        {
            W8P2();
        }

        static void W8P2()
        {
            Point[] training_points = Point.ImportPointSet( @"c:\features.train" );
            SVMHelper svm = new SVMHelper( training_points );
            svm._param.C = 0.1;
            svm._param.gamma = 1;
            svm._param.kernel_type = svm_parameter.POLY;
            svm._param.coef0 = 1;
            svm._param.degree = 2;

            svm.train();
            
        }
    }
}
