using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LearningFromData;
using System.IO;
using MathNet.Numerics.LinearAlgebra.Double;

namespace FinalExam
{
    class Program
    {
        static void Main( string[] args )
        {
            Random r = new Random();

            //int[] digits = new int[5] { 5, 6, 7, 8, 9 };
            //double[] ein = new double[digits.Count()];
            //double[] eout = new double[digits.Count()];
            //FEP7( r, digits, ein, eout, true );

            //int[] digits = new int[5] { 0, 1, 2, 3, 4 };
            //double[] ein = new double[digits.Count()];
            //double[] eout = new double[digits.Count()];
            //FEP8( r, digits, ein, eout, true );

            FEP9();
        }

        static void FEP9()
        {
            Random r = new Random();
            int[] digits = new int[10] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            double[] ein_x = new double[digits.Count()];
            double[] eout_x = new double[digits.Count()];
            double[] ein_z = new double[digits.Count()];
            double[] eout_z = new double[digits.Count()];
            FEP7( r, digits, ein_x, eout_x, false );
            FEP8( r, digits, ein_z, eout_z, false );

            //      Ein                     Eout
            // 0  : X 0.1234  Z 0.1234  --  X 0.1234  Z 0.1234

            Console.WriteLine( "      Ein                     Eout" );
            for( int i = 0; i < digits.Count(); i++ )
            {
                Console.WriteLine( digits[i].ToString() 
                    + "  : X "
                    + ein_x[i].ToString( "f4" )
                    + "  Z "
                    + eout_x[i].ToString( "f4" )
                    + "  --  X "
                    + ein_z[i].ToString( "f4" )
                    + "  Z "
                    + eout_z[i].ToString( "f4" ) );
            }

            Console.WriteLine( "digit, ein_x, eout_x, ein_z, eout_z" );
            for( int i = 0; i < digits.Count(); i++ )
            {
                Console.WriteLine( digits[i].ToString()
                    + ","
                    + ein_x[i].ToString( "f6" )
                    + ","
                    + eout_x[i].ToString( "f6" )
                    + ","
                    + ein_z[i].ToString( "f6" )
                    + ","
                    + eout_z[i].ToString( "f6" ) );
            }
            
            
            Console.ReadLine();
        }

        static void FEP8( Random r, int[] digits, double[] ein, double[] eout, bool print )
        {
            int d = 2;

            for( int i = 0; i < digits.Count(); i++ )
            {                
                List<double> y_train = new List<double>();
                List<double[]> raw_data = new List<double[]>();
                DataParseWithX0( @"c:\features.train", d, y_train, raw_data, digits[i] );
                Data[] training_points = RawToData( raw_data, y_train );

                List<double> y_test = new List<double>();
                raw_data = new List<double[]>();
                DataParseWithX0( @"c:\features.test", d, y_test, raw_data, digits[i] );
                Data[] test_points = RawToData( raw_data, y_test );

                // transform to Z : 1, x1, x2, x1 * x2, x1^2, x2^2
                DenseMatrix Z = new DenseMatrix( training_points.Count(), 6 );
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    Z[j, 0] = 1;
                    Z[j, 1] = training_points[j].x[1];
                    Z[j, 2] = training_points[j].x[2];
                    Z[j, 3] = training_points[j].x[1] * training_points[j].x[2];
                    Z[j, 4] = Math.Pow( training_points[j].x[1], 2 );
                    Z[j, 5] = Math.Pow( training_points[j].x[2], 2 );
                }

                DenseVector w = LearningTools.RunLinRegRegularized( training_points.Count(), Z, new DenseVector( y_train.ToArray() ),
                    1, r );

                // calculate Ein
                int num_wrong = 0;
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    if( Math.Sign( Z.Row( j ).DotProduct( w ) ) != training_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                ein[i] = (double)num_wrong / (double)training_points.Count();

                // calculate Eout
                num_wrong = 0;
                for( int j = 0; j < test_points.Count(); j++ )
                {
                    DenseVector z = new DenseVector( 6 );
                    z[0] = 1;
                    z[1] = test_points[j].x[1];
                    z[2] = test_points[j].x[2];
                    z[3] = test_points[j].x[1] * test_points[j].x[2];
                    z[4] = Math.Pow( test_points[j].x[1], 2 );
                    z[5] = Math.Pow( test_points[j].x[2], 2 );
                    if( Math.Sign( z.DotProduct( w ) ) != test_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                eout[i] = (double)num_wrong / (double)test_points.Count();
            }

            if( print )
            {
                Console.WriteLine( "--No Transform--" );
                for( int i = 0; i < digits.Count(); i++ )
                {
                    Console.WriteLine( "For " + digits[i].ToString() + " eout = " + eout[i].ToString( "f04" )
                        + " ein = " + ein[i].ToString( "f04" ) );
                }
                Console.ReadLine();
            }
        }

        static void FEP7( Random r, int[] digits, double[] ein, double[] eout, bool print )
        {
            int d = 2;

            for( int i = 0; i < digits.Count(); i++ )
            {
                List<double> y_train = new List<double>();
                List<double[]> raw_data = new List<double[]>();
                DataParseWithX0( @"c:\features.train", d, y_train, raw_data, digits[i] );
                Data[] training_points = RawToData( raw_data, y_train );

                List<double> y_test = new List<double>();
                raw_data = new List<double[]>();
                DataParseWithX0( @"c:\features.test", d, y_test, raw_data, digits[i] );
                Data[] test_points = RawToData( raw_data, y_test );

                DenseMatrix X = new DenseMatrix( training_points.Count(), d + 1 );
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    for( int k = 0; k <= d; k++ )
                    {
                        X[j, k] = training_points[j].x[k];
                    }
                }

                DenseVector w = LearningTools.RunLinRegRegularized( training_points.Count(), X, new DenseVector( y_train.ToArray() ),
                    1, r );

                // calculate Ein
                int num_wrong = 0;
                for( int j = 0; j < training_points.Count(); j++ )
                {
                    if( Math.Sign( X.Row( j ).DotProduct( w ) ) != training_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                ein[i] = (double)num_wrong / (double)training_points.Count();

                // calculate Eout
                num_wrong = 0;
                for( int j = 0; j < test_points.Count(); j++ )
                {
                    if( Math.Sign( test_points[j].AsDenseVector().DotProduct( w ) ) != test_points[j].y )
                    {
                        num_wrong++;
                    }
                }
                eout[i] = (double)num_wrong / (double)test_points.Count();
            }

            if( print )
            {
                Console.WriteLine( "--With Transform--" );
                for( int i = 0; i < digits.Count(); i++ )
                {
                    Console.WriteLine( "For " + digits[i].ToString() + " eout = " + eout[i].ToString( "f04" )
                        + " ein = " + ein[i].ToString( "f04" ) );
                }
                Console.ReadLine();
            }
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

        static void DataParseWithX0( string filename, int d, List<double> y, List<double[]> raw_data, int plusone )
        {
            string line;
            using( StreamReader file = new StreamReader( filename ) )
            {
                // if this was real programming, we'd put some error handling around this
                while( ( line = file.ReadLine() ) != null )
                {
                    double[] point = new double[d+1];
                    string[] parts = line.Split( new char[0], StringSplitOptions.RemoveEmptyEntries );
                    double yn = double.Parse( parts[0] );
                    point[0] = 1;
                    if( yn == plusone )
                    {
                        y.Add( 1 );
                        point[1] = double.Parse( parts[1] );
                        point[2] = double.Parse( parts[2] );
                        raw_data.Add( point );
                    }
                    else
                    {
                        y.Add( -1 );
                        point[1] = double.Parse( parts[1] );
                        point[2] = double.Parse( parts[2] );
                        raw_data.Add( point );
                    }
                }
                file.Close();
            }
        }
    }
}
