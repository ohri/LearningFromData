using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LearningFromData;
using libsvm;
using System.IO;

namespace Week8
{
    class W8
    {
        static void Main( string[] args )
        {
//            W8P234( new int[5] { 0, 2, 4, 6, 8 } );

//            W8P234( new int[5] { 1, 3, 5, 7, 9 } );

//            W8P56( new double[4] { .001, .01, .1, 1 }, 2 );

//            W8P56( new double[4] { .0001, .001, .01, 1 }, 2 );
//            W8P56( new double[4] { .0001, .001, .01, 1 }, 5 );

//            W8P7();
            W8P910( new double[5] { 0.01, 1, 100, 10000, 1000000 } );
        }

        static void W8P910( double[] Cs )
        {
            int d = 2;

            double[] ein = new double[Cs.Count()];
            int[] sv = new int[Cs.Count()];
            double[] eout = new double[Cs.Count()];

            List<double> y_train = new List<double>();
            List<double[]> raw_data = new List<double[]>();
            W8P56DataParse( @"c:\features.train", d, y_train, raw_data );
            Data[] training_points = RawToData( raw_data, y_train );

            List<double> y_test = new List<double>();
            raw_data = new List<double[]>();
            W8P56DataParse( @"c:\features.test", d, y_test, raw_data );
            Data[] test_points = RawToData( raw_data, y_test );

            for( int i = 0; i < Cs.Count(); i++ )
            {
                SVMHelper svm = new SVMHelper( training_points );
                svm.prob.y = y_train.ToArray();
                svm.prob.l = y_train.Count();

                svm.param.C = Cs[i];
                svm.param.gamma = 1;
                svm.param.kernel_type = svm_parameter.RBF;

                svm.train();
                sv[i] = svm.model.SV.Count();

                // grade the run for Ein
                //int fail = 0;
                //for( int j = 0; j < training_points.Count(); j++ )
                //{
                //    int svm_y = Math.Sign( training_points[j].AsDenseVector().DotProduct( svm.w ) + svm.b );
                //    if( svm_y != y_train[j] )
                //    {
                //        fail++;
                //    }
                //}
                //ein[i] = (double)fail / (double)training_points.Count();
                ein[i] = svm.predict( training_points );

                // grade the run for Eout
                //fail = 0;
                //for( int j = 0; j < test_points.Count(); j++ )
                //{
                //    int svm_y = Math.Sign( test_points[j].AsDenseVector().DotProduct( svm.w ) + svm.b );
                //    if( svm_y != y_test[j] )
                //    {
                //        fail++;
                //    }
                //}
                //eout[i] = (double)fail / (double)test_points.Count();
                eout[i] = svm.predict( test_points );
            }

            Console.WriteLine( "" );
            Console.WriteLine( "" );
            for( int i = 0; i < Cs.Count(); i++ )
            {
                Console.WriteLine( "For C=" + Cs[i].ToString() + " : Ein is " + ein[i].ToString()
                    + " -- Eout is " + eout[i].ToString() + " -- " + sv[i].ToString() + " support vectors" );
            }
            Console.ReadLine();
        }

        static void W8P7()
        {
            int d = 2;
            int runs = 100;
            double[] Cs = new double[5] { .0001, .001, .01, .1, 1 };
            double[] Ecv = new double[Cs.Count()];
            int[] Cwins = new int[Cs.Count()];

            List<double> y_train = new List<double>();
            List<double[]> raw_data = new List<double[]>();
            W8P56DataParse( @"c:\features.train", d, y_train, raw_data );
            Data[] training_points = RawToData( raw_data, y_train );

            for( int i = 0; i < runs; i++ )
            {
                double[] Ecv_run = new double[Cs.Count()];
                for( int j = 0; j < Cs.Count(); j++ )
                {
                    SVMHelper svm = new SVMHelper( training_points );
                    svm.prob.y = y_train.ToArray();
                    svm.prob.l = y_train.Count();

                    svm.param.C = Cs[j];
                    svm.param.gamma = 1;
                    svm.param.kernel_type = svm_parameter.POLY;
                    svm.param.coef0 = 1;
                    svm.param.degree = 2;

                    Ecv_run[j] = svm.cross_validation( 10 );
                    Ecv[j] += Ecv_run[j];
                }

                int winner = FindLowestPercentage( Ecv_run );
                Cwins[winner]++;
            }

            Console.WriteLine( "" );
            Console.WriteLine( "" );
            for( int i = 0; i < Cs.Count(); i++ )
            {
                Ecv[i] = Ecv[i] / (double)runs;
                Console.WriteLine( "For C=" + Cs[i].ToString() + " got " + Cwins[i].ToString() + " wins, avg Ecv of "
                    + Ecv[i].ToString() );
            }
            Console.ReadLine();
        }

        static int FindLowestPercentage( double[] vals )
        {
            int index = -1;
            double lowest = 1.01;
            for( int i = 0; i < vals.Count(); i++ )
            {
                if( vals[i] < lowest )
                {
                    lowest = vals[i];
                    index = i;
                }
            }
            return index;
        }

        static void W8P56( double[] Cs, int Q )
        {
            int d = 2;

            double[] ein = new double[Cs.Count()];
            int[] sv = new int[Cs.Count()];
            double[] eout = new double[Cs.Count()];

            List<double> y_train = new List<double>();
            List<double[]> raw_data = new List<double[]>();
            W8P56DataParse( @"c:\features.train", d, y_train, raw_data );
            Data[] training_points = RawToData( raw_data, y_train );

            List<double> y_test = new List<double>();
            raw_data = new List<double[]>();
            W8P56DataParse( @"c:\features.test", d, y_test, raw_data );
            Data[] test_points = RawToData( raw_data, y_test );

            for( int i = 0; i < Cs.Count(); i++ )
            {
                SVMHelper svm = new SVMHelper( training_points );
                svm.prob.y = y_train.ToArray();
                svm.prob.l = y_train.Count();

                svm.param.C = Cs[i];
                svm.param.gamma = 1;
                svm.param.kernel_type = svm_parameter.POLY;
                svm.param.coef0 = 1;
                svm.param.degree = Q;

                svm.train();
                sv[i] = svm.model.SV.Count();

                // grade the run for Ein
                int fail = 0;
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    int svm_y = Math.Sign( training_points[j].AsDenseVector().DotProduct( svm.w ) + svm.b );
                    if( svm_y != y_train[j] )
                    {
                        fail++;
                    }
                }
                ein[i] = (double)fail / (double)training_points.Count();

                // grade the run for Eout
                fail = 0;
                for( int j = 0; j < test_points.Count(); j++ )
                {
                    int svm_y = Math.Sign( test_points[j].AsDenseVector().DotProduct( svm.w ) + svm.b );
                    if( svm_y != y_test[j] )
                    {
                        fail++;
                    }
                }
                eout[i] = (double)fail / (double)test_points.Count();
            }

            Console.WriteLine( "" );
            Console.WriteLine( "For Q = " + Q.ToString() );
            for( int i = 0; i < Cs.Count(); i++ )
            {
                Console.WriteLine( "For C=" + Cs[i].ToString() + " : Ein is " + ein[i].ToString() 
                    + " -- Eout is " + eout[i].ToString() + " -- " + sv[i].ToString() + " support vectors" );
            }
            Console.ReadLine();
        }

        private static Data[] RawToData( List<double[]> raw_data, List<double> y )
        {
            Data[] training_points = new Data[raw_data.Count()];
            for( int j = 0; j < raw_data.Count(); j++ )
            {
                training_points[j] = new Data( raw_data[j] );
                training_points[j].y = Math.Sign( y[j] );
            }
            return training_points;
        }

        static void W8P56DataParse( string filename, int d, List<double> y, List<double[]> raw_data )
        {
            string line;
            using( StreamReader file = new StreamReader( filename ) )
            {
                // if this was real programming, we'd put some error handling around this
                while( ( line = file.ReadLine() ) != null )
                {
                    double[] point = new double[d];
                    string[] parts = line.Split( new char[0], StringSplitOptions.RemoveEmptyEntries );
                    double yn = double.Parse( parts[0] );
                    if( yn == 1 )
                    {
                        y.Add( 1 );
                        point[0] = double.Parse( parts[1] );
                        point[1] = double.Parse( parts[2] );
                        raw_data.Add( point );
                    }
                    else if( yn == 5 )
                    {
                        y.Add( -1 );
                        point[0] = double.Parse( parts[1] );
                        point[1] = double.Parse( parts[2] );
                        raw_data.Add( point );
                    }
                }
                file.Close();
            }
        }

        static void W8P234( int[] do_these )
        {
            int d = 2;

            double[] ein = new double[do_these.Count()];
            int[] sv = new int[do_these.Count()];

            for( int i = 0; i < do_these.Count(); i++ )
            {
                List<double[]> raw_data = new List<double[]>();
                List<double> y = new List<double>();
                string line;
                using( StreamReader file = new StreamReader( @"c:\features.train" ) )
                {
                    // if this was real programming, we'd put some error handling around this
                    while( ( line = file.ReadLine() ) != null )
                    {
                        double[] point = new double[d];
                        string[] parts = line.Split( new char[0], StringSplitOptions.RemoveEmptyEntries );
                        double yn = double.Parse( parts[0] );
                        if( yn == do_these[i] )
                        {
                            y.Add( 1 );
                        }
                        else
                        {
                            y.Add( -1 );
                        }
                        point[0] = double.Parse( parts[1] );
                        point[1] = double.Parse( parts[2] );
                        raw_data.Add( point );
                    }
                    file.Close();
                }

                Data[] training_points = new Data[raw_data.Count()];
                for( int j = 0; j < raw_data.Count(); j++ )
                {
                    training_points[j] = new Data( raw_data[j] );
                }

                SVMHelper svm = new SVMHelper( training_points );
                svm.prob.y = y.ToArray();
                svm.prob.l = y.Count();

                svm.param.C = 0.01;
                svm.param.gamma = 1;
                svm.param.kernel_type = svm_parameter.POLY;
                svm.param.coef0 = 1;
                svm.param.degree = 2;

                svm.train();

                sv[i] = svm.model.SV.Count();

                // now grade the run for Ein
                int fail = 0;
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    int svm_y = Math.Sign( training_points[j].AsDenseVector().DotProduct( svm.w ) + svm.b );
                    if( svm_y != y[j] )
                    {
                        fail++;
                    }
                }
                ein[i] = (double)fail / (double)training_points.Count();
            }

            Console.WriteLine( "" );
            Console.WriteLine( "" );
            for( int i = 0; i < do_these.Count(); i++ )
            {
                Console.WriteLine( "Ein for digit " + do_these[i].ToString() + " was " + ein[i].ToString()
                    + " and it had " + sv[i].ToString() + " support vectors" );
            }
            Console.ReadLine();
        }
    }
}
