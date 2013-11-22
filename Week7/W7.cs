using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LearningFromData;
using MathNet.Numerics.LinearAlgebra.Double;
using libsvm;

namespace Week7
{
    class W7
    {
        static void Main( string[] args )
        {
//            W7P1();
//            W7P2();
//            W7P3();
//            W7P4();
//            W7P6();
            W7P8();
        }

        public static void W7P1()
        {
            Point[] data_points = Point.ImportPointSet( @"c:\LFD_W6_P2_insample.txt" );

            // split this into training points (first 25) and validation points (last 10)
            Point[] training_points = data_points.Take( 25 ).ToArray();
            Point[] validation_points = data_points.Skip( 25 ).Take( 10 ).ToArray();

            int k = 2;

            // transform these to Z
            DenseMatrix X = new DenseMatrix( training_points.Count(), k+1 );
            DenseVector y = new DenseVector( training_points.Count() );
            for( int i = 0; i < training_points.Count(); i++ )
            {
                X[i, 0] = 1;
                if( k >= 1 )
                    X[i, 1] = training_points[i].x;
                if( k >= 2 )
                    X[i, 2] = training_points[i].y;
                if( k >= 3 )
                    X[i, 3] = Math.Pow( training_points[i].x, 2 );
                if( k >= 4 )
                    X[i, 4] = Math.Pow( training_points[i].y, 2 );
                if( k >= 5 )
                    X[i, 5] = training_points[i].x * training_points[i].y;
                if( k >= 6 )
                    X[i, 6] = Math.Abs( training_points[i].x - training_points[i].y );
                if( k >= 7 )
                    X[i, 7] = Math.Abs( training_points[i].x + training_points[i].y );

                y[i] = training_points[i].fx;
            }

            Random r = new Random();
            DenseVector w = LearningTools.RunLinearRegression( training_points.Count(), X, y, r );

            // now calculate in sample error
            int num_wrong = 0;
            for( int i = 0; i < training_points.Count(); i++ )
            {
                if( Math.Sign( X.Row( i ).DotProduct( w ) ) != training_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double ein = (double)num_wrong / (double)training_points.Count();

            // and out of sample error
            num_wrong = 0;
            for( int i = 0; i < validation_points.Count(); i++ )
            {
                DenseVector x = new DenseVector( k + 1 );
                x[0] = 1;
                if( k >= 1 )
                    x[1] = validation_points[i].x;
                if( k >= 2 )
                    x[2] = validation_points[i].y;
                if( k >= 3 )
                    x[3] = Math.Pow( validation_points[i].x, 2 );
                if( k >= 4 )
                    x[4] = Math.Pow( validation_points[i].y, 2 );
                if( k >= 5 )
                    x[5] = validation_points[i].x * validation_points[i].y;
                if( k >= 6 )
                    x[6] = Math.Abs( validation_points[i].x - validation_points[i].y );
                if( k >= 7 )
                    x[7] = Math.Abs( validation_points[i].x + validation_points[i].y );

                if( Math.Sign( x.DotProduct( w ) ) != validation_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double eout = (double)num_wrong / (double)validation_points.Count();

            Console.WriteLine( "With k of " + k.ToString() + ", Ein is " + ein.ToString() + " and Eout (using validation points) is " + eout.ToString() );
            string dc = Console.ReadLine();
        }

        public static void W7P2()
        {
            Point[] data_points = Point.ImportPointSet( @"c:\LFD_W6_P2_insample.txt" );
            Point[] out_points = Point.ImportPointSet( @"c:\LFD_W6_P2_outsample.txt" );

            // split this into training points (first 25) 
            Point[] training_points = data_points.Take( 25 ).ToArray();

            int k = 3;

            // transform these to Z
            DenseMatrix X = new DenseMatrix( training_points.Count(), k + 1 );
            DenseVector y = new DenseVector( training_points.Count() );
            for( int i = 0; i < training_points.Count(); i++ )
            {
                X[i, 0] = 1;
                if( k >= 1 )
                    X[i, 1] = training_points[i].x;
                if( k >= 2 )
                    X[i, 2] = training_points[i].y;
                if( k >= 3 )
                    X[i, 3] = Math.Pow( training_points[i].x, 2 );
                if( k >= 4 )
                    X[i, 4] = Math.Pow( training_points[i].y, 2 );
                if( k >= 5 )
                    X[i, 5] = training_points[i].x * training_points[i].y;
                if( k >= 6 )
                    X[i, 6] = Math.Abs( training_points[i].x - training_points[i].y );
                if( k >= 7 )
                    X[i, 7] = Math.Abs( training_points[i].x + training_points[i].y );

                y[i] = training_points[i].fx;
            }

            Random r = new Random();
            DenseVector w = LearningTools.RunLinearRegression( training_points.Count(), X, y, r );

            // now calculate in sample error
            int num_wrong = 0;
            for( int i = 0; i < training_points.Count(); i++ )
            {
                if( Math.Sign( X.Row( i ).DotProduct( w ) ) != training_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double ein = (double)num_wrong / (double)training_points.Count();

            // and out of sample error
            num_wrong = 0;
            for( int i = 0; i < out_points.Count(); i++ )
            {
                DenseVector x = new DenseVector( k + 1 );
                x[0] = 1;
                if( k >= 1 )
                    x[1] = out_points[i].x;
                if( k >= 2 )
                    x[2] = out_points[i].y;
                if( k >= 3 )
                    x[3] = Math.Pow( out_points[i].x, 2 );
                if( k >= 4 )
                    x[4] = Math.Pow( out_points[i].y, 2 );
                if( k >= 5 )
                    x[5] = out_points[i].x * out_points[i].y;
                if( k >= 6 )
                    x[6] = Math.Abs( out_points[i].x - out_points[i].y );
                if( k >= 7 )
                    x[7] = Math.Abs( out_points[i].x + out_points[i].y );

                if( Math.Sign( x.DotProduct( w ) ) != out_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double eout = (double)num_wrong / (double)out_points.Count();

            Console.WriteLine( "With k of " + k.ToString() + ", Ein is " + ein.ToString() + " and Eout (using out data) is " + eout.ToString() );
            string dc = Console.ReadLine();
        }

        public static void W7P3()
        {
            Point[] data_points = Point.ImportPointSet( @"c:\LFD_W6_P2_insample.txt" );

            // split this into training points (last 10) and validation points (first 25)
            Point[] validation_points = data_points.Take( 25 ).ToArray();
            Point[] training_points = data_points.Skip( 25 ).Take( 10 ).ToArray();

            int k = 6;

            // transform these to Z
            DenseMatrix X = new DenseMatrix( training_points.Count(), k + 1 );
            DenseVector y = new DenseVector( training_points.Count() );
            for( int i = 0; i < training_points.Count(); i++ )
            {
                X[i, 0] = 1;
                if( k >= 1 )
                    X[i, 1] = training_points[i].x;
                if( k >= 2 )
                    X[i, 2] = training_points[i].y;
                if( k >= 3 )
                    X[i, 3] = Math.Pow( training_points[i].x, 2 );
                if( k >= 4 )
                    X[i, 4] = Math.Pow( training_points[i].y, 2 );
                if( k >= 5 )
                    X[i, 5] = training_points[i].x * training_points[i].y;
                if( k >= 6 )
                    X[i, 6] = Math.Abs( training_points[i].x - training_points[i].y );
                if( k >= 7 )
                    X[i, 7] = Math.Abs( training_points[i].x + training_points[i].y );

                y[i] = training_points[i].fx;
            }

            Random r = new Random();
            DenseVector w = LearningTools.RunLinearRegression( training_points.Count(), X, y, r );

            // now calculate in sample error
            int num_wrong = 0;
            for( int i = 0; i < training_points.Count(); i++ )
            {
                if( Math.Sign( X.Row( i ).DotProduct( w ) ) != training_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double ein = (double)num_wrong / (double)training_points.Count();

            // and out of sample error
            num_wrong = 0;
            for( int i = 0; i < validation_points.Count(); i++ )
            {
                DenseVector x = new DenseVector( k + 1 );
                x[0] = 1;
                if( k >= 1 )
                    x[1] = validation_points[i].x;
                if( k >= 2 )
                    x[2] = validation_points[i].y;
                if( k >= 3 )
                    x[3] = Math.Pow( validation_points[i].x, 2 );
                if( k >= 4 )
                    x[4] = Math.Pow( validation_points[i].y, 2 );
                if( k >= 5 )
                    x[5] = validation_points[i].x * validation_points[i].y;
                if( k >= 6 )
                    x[6] = Math.Abs( validation_points[i].x - validation_points[i].y );
                if( k >= 7 )
                    x[7] = Math.Abs( validation_points[i].x + validation_points[i].y );

                if( Math.Sign( x.DotProduct( w ) ) != validation_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double eout = (double)num_wrong / (double)validation_points.Count();

            Console.WriteLine( "With k of " + k.ToString() + ", Ein is " + ein.ToString() + " and Eout (using validation points) is " + eout.ToString() );
            string dc = Console.ReadLine();
        }

        public static void W7P4()
        {
            Point[] data_points = Point.ImportPointSet( @"c:\LFD_W6_P2_insample.txt" );
            Point[] out_points = Point.ImportPointSet( @"c:\LFD_W6_P2_outsample.txt" );

            // split this into training points (last 10) 
            Point[] training_points = data_points.Skip( 25 ).Take( 10 ).ToArray();

            int k = 7;

            // transform these to Z
            DenseMatrix X = new DenseMatrix( training_points.Count(), k + 1 );
            DenseVector y = new DenseVector( training_points.Count() );
            for( int i = 0; i < training_points.Count(); i++ )
            {
                X[i, 0] = 1;
                if( k >= 1 )
                    X[i, 1] = training_points[i].x;
                if( k >= 2 )
                    X[i, 2] = training_points[i].y;
                if( k >= 3 )
                    X[i, 3] = Math.Pow( training_points[i].x, 2 );
                if( k >= 4 )
                    X[i, 4] = Math.Pow( training_points[i].y, 2 );
                if( k >= 5 )
                    X[i, 5] = training_points[i].x * training_points[i].y;
                if( k >= 6 )
                    X[i, 6] = Math.Abs( training_points[i].x - training_points[i].y );
                if( k >= 7 )
                    X[i, 7] = Math.Abs( training_points[i].x + training_points[i].y );

                y[i] = training_points[i].fx;
            }

            Random r = new Random();
            DenseVector w = LearningTools.RunLinearRegression( training_points.Count(), X, y, r );

            // now calculate in sample error
            int num_wrong = 0;
            for( int i = 0; i < training_points.Count(); i++ )
            {
                if( Math.Sign( X.Row( i ).DotProduct( w ) ) != training_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double ein = (double)num_wrong / (double)training_points.Count();

            // and out of sample error
            num_wrong = 0;
            for( int i = 0; i < out_points.Count(); i++ )
            {
                DenseVector x = new DenseVector( k + 1 );
                x[0] = 1;
                if( k >= 1 )
                    x[1] = out_points[i].x;
                if( k >= 2 )
                    x[2] = out_points[i].y;
                if( k >= 3 )
                    x[3] = Math.Pow( out_points[i].x, 2 );
                if( k >= 4 )
                    x[4] = Math.Pow( out_points[i].y, 2 );
                if( k >= 5 )
                    x[5] = out_points[i].x * out_points[i].y;
                if( k >= 6 )
                    x[6] = Math.Abs( out_points[i].x - out_points[i].y );
                if( k >= 7 )
                    x[7] = Math.Abs( out_points[i].x + out_points[i].y );

                if( Math.Sign( x.DotProduct( w ) ) != out_points[i].fx )
                {
                    num_wrong++;
                }
            }
            double eout = (double)num_wrong / (double)out_points.Count();

            Console.WriteLine( "With k of " + k.ToString() + ", Ein is " + ein.ToString() + " and Eout (using out data) is " + eout.ToString() );
            string dc = Console.ReadLine();
        }

        public static void W7P6()
        {
            Random r = new Random();
            int runs = 1000000;

            double avg_e1 = 0;
            double avg_e2 = 0;
            double avg_e = 0;

            for( int i = 0; i < runs; i++ )
            {
                double e1 = r.NextDouble();
                double e2 = r.NextDouble();
                double e = Math.Min( e1, e2 );

                avg_e1 += e1;
                avg_e2 += e2;
                avg_e += e;
            }

            avg_e = avg_e / (double)runs;
            avg_e1 = avg_e1 / (double)runs;
            avg_e2 = avg_e2 / (double)runs;

            Console.WriteLine( "Expected e1 is " + avg_e1.ToString() + ", e2 is " + avg_e2.ToString() + ", e is " + avg_e.ToString() );
            string dc = Console.ReadLine();
        }

        public static void W7P8()
        {
            Random r = new Random();
            int n = 100;
            int n_test = 5000;
            int runs = 1000;

            int pla_fails = 0;
            int svm_fails = 0;
            int pla_wins = 0;
            int svm_wins = 0;
            int total_sv = 0;

            for( int i = 0; i < runs; i++ )
            {
                int pla_fails_this_run = 0;
                int svm_fails_this_run = 0;

                Line l = Line.CreateRandomLine( r );                
                Point[] training_points = null;
                bool good_to_go = false;

                // make sure not all points are on the same side of the line
                while( !good_to_go )
                {
                    int sum_of_y = 0;
                    training_points = Point.CreatePointSet( n, l, r );
                    foreach( Point p in training_points )
                    {
                        sum_of_y += p.fx;
                    }
                    if( sum_of_y < n && sum_of_y > ( n * -1 ) )
                    {
                        good_to_go = true;
                    }
                }

                // train the PLA
                DenseVector w = new DenseVector( 3 );
                int dc = LearningTools.RunPerceptron( n, training_points, w, r );

                // test the PLA with test points
                Point[] test_points = Point.CreatePointSet( n_test, l, r );
                pla_fails_this_run += LearningTools.TestPoints( w, test_points );

                SVMHelper svm = new SVMHelper( training_points );
                svm.train();

                // about 10% of my runs are coming through with no support vectors!?
                if( svm._model.SV.Count() > 0 )
                {
                    total_sv += svm._model.SV.Count();

                    // W is the sum of alpha * xn * yn for each of the support vectors
                    DenseVector wsvm = new DenseVector( 2 );
                    for( int j = 0; j < svm._model.SV.Count(); j++ )
                    {
                        int yn = training_points[find_x( svm._model.SV[j][0].value_Renamed, svm._model.SV[j][1].value_Renamed, training_points )].fx;

                        DenseVector xn = new DenseVector( 2 );
                        xn[0] = svm._model.SV[j][0].value_Renamed;
                        xn[1] = svm._model.SV[j][1].value_Renamed;

                        xn.Multiply( (double)yn * svm._model.sv_coef[0][j] );
                        wsvm = (DenseVector)wsvm.Add( xn );
                    }

                    // calculate b = 1/yn - wsvm_transpose * xn for any support vector
                    DenseVector xsv = new DenseVector( 2 );
                    xsv[0] = svm._model.SV[0][0].value_Renamed;
                    xsv[1] = svm._model.SV[0][1].value_Renamed;
                    int y_sv = training_points[find_x( xsv[0], xsv[1], training_points )].fx;
                    double b = 1.0/(double)y_sv - xsv.DotProduct( wsvm );

                    // test SVM with test points
                    for( int j = 0; j < test_points.Count(); j++ )
                    {
                        DenseVector xn = new DenseVector( 2 );
                        xn[0] = test_points[j].x;
                        xn[1] = test_points[j].y;
                        int svm_y = Math.Sign( xn.DotProduct( wsvm ) + b );
                        if( svm_y != test_points[j].fx )
                        {
                            svm_fails_this_run++;
                        }
                    }
                    if( svm_fails_this_run < pla_fails_this_run )
                    {
                        svm_wins++;
                    }
                    else if( svm_fails_this_run > pla_fails_this_run )
                    {
                        pla_wins++;
                    }
                    svm_fails += svm_fails_this_run;
                    pla_fails += pla_fails_this_run;
                }
            }

            double pla_fail_rate = (double)pla_fails / (double)runs / (double)n_test;
            double svm_fail_rate = (double)svm_fails / (double)runs / (double)n_test;
            double avg_sv = (double)total_sv / (double)runs;
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine( "Perceptron had average of " + pla_fail_rate.ToString() + " Eout." );
            Console.WriteLine( "SVM had average of " + svm_fail_rate.ToString() + " Eout." );
            Console.WriteLine( "SVM won " + svm_wins.ToString() + ", PLA won " + pla_wins.ToString() );
            Console.WriteLine( "Average # of SV was " + avg_sv.ToString() );
            Console.ReadLine();
        }

        public static int find_x( double x, double y, Point[] points )
        {
            int i = 0;
            while( i < points.Count() )
            {
                if( points[i].x == x && points[i].y == y  )
                {
                    return i;
                }
                i++;
            }
            return -1;
        }
    }
}
