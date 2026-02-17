import unittest
from unittest.mock import patch, mock_open, Mock
import os
import shutil
import tempfile
from pathlib import Path
from obsidian_to_jekyll import *

class TestObsidianToJekyll(unittest.TestCase):
    def setUp(self):
        
        # Create temporary directories for testing
        self.obsidian_root = Path(tempfile.mkdtemp())
        
        self.obsidian_publish_dir = self.obsidian_root / "Publish"
        self.obsidian_img_dir = self.obsidian_root / "Assets" / "Images"
        self.jekyll_root = self.obsidian_root / "schrodlm.github.io"
        self.jekyll_img_dir = self.jekyll_root / "assets" / "img"

        ObsidianPath.configure_root(self.obsidian_root)
        JekyllPath.configure_root(self.jekyll_root)

        self.obsidian_publish_dir.mkdir(parents=True, exist_ok=True)
        self.obsidian_img_dir.mkdir(parents=True, exist_ok=True)
        self.jekyll_root.mkdir(parents=True, exist_ok=True)
        self.jekyll_img_dir.mkdir(parents=True, exist_ok=True)

        # Create jekyll subdirectory
        self.jekyll_subdir1 = self.jekyll_root / "_posts"
        self.jekyll_subdir2 = self.jekyll_root / "_projects"
        self.jekyll_subdir1.mkdir(parents=True, exist_ok=True)
        self.jekyll_subdir2.mkdir(parents=True, exist_ok=True)

        # Create publish subdirectories
        self.obsidian_subdir1 = self.obsidian_publish_dir / "Posts"
        self.obsidian_subdir2 = self.obsidian_publish_dir / "Projects"
        self.obsidian_subdir1.mkdir(parents=True, exist_ok=True)
        self.obsidian_subdir2.mkdir(parents=True, exist_ok=True)                    

        # Create a test image
        self.fake_image = self.obsidian_img_dir / "test-image.png"
        self.fake_image.write_text("""fake image data""")

        # Create test files with Obsidian links
        # File 1: References file 2 and has an image
        self.file1 = self.obsidian_subdir1 / "file1.md"
        self.file1.write_text("""# File 1
            
This file references [[file2| reference]], [[file2 | reference]], [[file2 |reference]] and has an image:

![test-image.png](test-image.png)

Also references [[file2#chapter]].
""")
        
        # File 2: In a different subdirectory, references file 1
        self.file2 = self.obsidian_subdir2 / "file2.md"
        self.file2.write_text("""# File 2
        This is file 2.
""")
          
    def tearDown(self):
        # Clean up temporary directories
        shutil.rmtree(self.obsidian_root)
        pass

    def test_path_validation(self):
            """Test the ObsidianPath and JekyllPath validation."""
            # Configure the paths for testing
            with patch('obsidian_to_jekyll.OBSIDIAN_ROOT', self.obsidian_root), \
                patch('obsidian_to_jekyll.JEKYLL_ROOT', self.jekyll_root):
                
                # --- Valid Paths ---
                with self.subTest("Valid ObsidianPath"):
                    valid_obsidian = ObsidianPath(self.obsidian_subdir1 / "file1.md")
                    self.assertTrue(valid_obsidian.exists())
                    
                with self.subTest("Valid JekyllPath"):
                    valid_jekyll = JekyllPath(self.jekyll_subdir1 / "post.md")
                    self.assertTrue(valid_jekyll.parent.exists())
                
                # --- Invalid Paths ---
                with self.subTest("Invalid ObsidianPath"):
                    with self.assertRaises(ValueError):
                        ObsidianPath(Path(tempfile.mkdtemp()))
                        
                with self.subTest("Invalid JekyllPath"):
                    with self.assertRaises(ValueError):
                        JekyllPath(self.obsidian_publish_dir / "outside_jekyll.md")

    def test_get_jekyll_directory(self):
        with self.subTest("Valid directory test"):
            got = self.obsidian_subdir1
            want = self.jekyll_subdir1
            self.assertEqual(want,get_jekyll_directory(got, self.jekyll_root, self.obsidian_publish_dir))

        with self.subTest("Not created in Jekyll dir structure"):
            got = self.obsidian_publish_dir / "DoesNotExistsInJekyll"
            with self.assertRaises(RuntimeError):
                get_jekyll_directory(got, self.jekyll_root, self.obsidian_publish_dir)
        
        with self.subTest("Invalid parent"):
            got = self.obsidian_img_dir / "Posts"
            with self.assertRaises(RuntimeError):
                get_jekyll_directory(got, self.jekyll_root, self.obsidian_publish_dir)

    def test_get_publish_subdirectories(self):
        with patch.object(Path, 'iterdir', return_value=[Path("subdir1"), Path("subdir2"), Path("subdir3")]):
            with self.subTest("Valid subdirectories"), patch.object(Path, 'is_dir', return_value=True):
                    subdirectories = get_publish_subdirectories(self.obsidian_publish_dir)
                    self.assertEqual(subdirectories, [Path('subdir1'), Path('subdir2'), Path('subdir3')], "Listed publish subdirectories are not correct.")
            
            with self.subTest("Publish dir contains files"), self.assertRaises(RuntimeError):
                get_publish_subdirectories(self.obsidian_publish_dir)
        
        with self.subTest("Unmocked"):
            subdirectories = get_publish_subdirectories(self.obsidian_publish_dir)
            self.assertEqual(subdirectories, [self.obsidian_subdir1, self.obsidian_subdir2], "Listed publish subdirectories are not correct.")
    
    def test_remove_contents_of(self):
        # This function has a safety check that depends on a global constant,
        # OBSIDIAN_DIR. We use `patch` to temporarily set this constant
        # to our main test directory for the duration of this test.
        with patch('obsidian_to_jekyll.JEKYLL_ROOT', self.obsidian_root):
            
            # --- Sub-test 1: Successful removal of contents ---
            with self.subTest("Successful removal of contents"):
                # Arrange: Create a directory structure to be deleted
                target_dir = Path(self.jekyll_subdir1 / "test_removal_directory")
                target_dir.mkdir()
                test_file = target_dir / "file_to_delete.md"
                test_file.write_text("some content")
                nested_dir = target_dir / "nested_dir"
                nested_dir.mkdir()
                nested_file = nested_dir / "nested_file.txt"
                nested_file.write_text("more content")

                # Act: Call the function to clear the directory
                remove_contents_of(target_dir)

                # Assert: The directory should still exist, but be empty
                self.assertTrue(target_dir.exists())
                self.assertEqual(list(target_dir.iterdir()), [], "Directory should be empty after removal")
                #Clean up
                shutil.rmtree(target_dir)

            
            # --- Sub-test 2: Safety check to prevent dangerous deletion ---
            with self.subTest("Safety check raises error for outside directory"):
                # Arrange: Create a separate temp directory outside the "safe" zone
                outside_dir = Path(tempfile.mkdtemp())
                # Add a file to it to ensure it's not empty
                (outside_dir / "do_not_delete.txt").touch()

                # Act & Assert: Expect a RuntimeError when trying to delete outside the safe zone
                with self.assertRaisesRegex(RuntimeError, "Trying to remove contents outside of this project! Aborted."):
                    remove_contents_of(outside_dir)

                # Assert that the outside directory and its contents were NOT deleted
                self.assertTrue((outside_dir / "do_not_delete.txt").exists())

                # Clean up
                shutil.rmtree(outside_dir)


    def test_get_directory_md_files(self):
        # Test 1: Basic functionality - finds markdown files in root directory
        with self.subTest("Finds markdown files in root directory"):
            # Arrange
            md_file1 = self.obsidian_publish_dir / "note1.md"
            md_file2 = self.obsidian_publish_dir / "note2.md"
            txt_file = self.obsidian_publish_dir / "text.txt"
            md_file1.touch()
            md_file2.touch()
            txt_file.touch()
            
            # Act
            result = get_directory_md_files(self.obsidian_publish_dir)
            
            # Assert
            self.assertEqual(len(result), 2)
            self.assertIn(md_file1, result)
            self.assertIn(md_file2, result)
            self.assertNotIn(txt_file, result)
            
            # Cleanup
            md_file1.unlink()
            md_file2.unlink()
            txt_file.unlink()

        # Test 2: Empty directory returns empty list
        with self.subTest("Empty directory returns empty list"):
            # Arrange
            empty_dir = self.obsidian_publish_dir / "empty_dir"
            empty_dir.mkdir()
            
            # Act
            result = get_directory_md_files(empty_dir)
            
            # Assert
            self.assertEqual(result, [])
            
            # Cleanup
            empty_dir.rmdir()

        # Test 3: No markdown files returns empty list
        with self.subTest("No markdown files returns empty list"):
            # Arrange
            no_md_dir = self.obsidian_publish_dir / "no_md_dir"
            no_md_dir.mkdir()
            txt_file = no_md_dir / "file.txt"
            csv_file = no_md_dir / "data.csv"
            txt_file.touch()
            csv_file.touch()
            
            # Act
            result = get_directory_md_files(no_md_dir)
            
            # Assert
            self.assertEqual(result, [])
            
            # Cleanup
            txt_file.unlink()
            csv_file.unlink()
            no_md_dir.rmdir()

        # Test 4: Should not follow symlinks
        with self.subTest("Should not follow symlinks"):
            # Arrange
            real_md = self.obsidian_publish_dir / "real.md"
            symlink_md = self.obsidian_publish_dir / "symlink.md"
            real_md.touch()
            symlink_md.symlink_to(real_md)
            
            # Act
            result = get_directory_md_files(self.obsidian_publish_dir)
            
            # Assert
            self.assertEqual(len(result), 2)  # Both the real file and symlink should be found
            self.assertIn(real_md, result)
            self.assertIn(symlink_md, result)
            
            # Cleanup
            real_md.unlink()
            symlink_md.unlink()

    def test_read_md_metadata(self):
        """Test reading metadata from markdown front matter"""
        # Test 1: Normal front matter
        with self.subTest("Normal front matter"):
            # Arrange
            md_file = self.obsidian_publish_dir / "normal_frontmatter.md"
            md_file.write_text("""---
Title: Test Document
Author: John Doe
Date: 2023-01-01
---
# Content""")
            
            # Act
            result = read_md_metadata(md_file)
            
            # Assert
            self.assertEqual(result, {
                'Title': 'Test Document',
                'Author': 'John Doe',
                'Date': '2023-01-01'
            })
            
            # Cleanup
            md_file.unlink()

        # Test 2: Unclosed front matter
        with self.subTest("Unclosed front matter"):
            # Arrange
            md_file = self.obsidian_publish_dir / "unclosed_frontmatter.md"
            md_file.write_text("""---
Title: Unclosed
Author: Test
# Content""")
            
            # Act
            result = read_md_metadata(md_file)
            
            # Assert
            self.assertEqual(result, {})
            
            # Cleanup
            md_file.unlink()

        # Test 3: No front matter
        with self.subTest("No front matter"):
            # Arrange
            md_file = self.obsidian_publish_dir / "no_frontmatter.md"
            md_file.write_text("# Just regular content")
            
            # Act
            result = read_md_metadata(md_file)
            
            # Assert
            self.assertEqual(result, {})
            
            # Cleanup
            md_file.unlink()

        # Test 4: Empty front matter
        with self.subTest("Empty front matter"):
            # Arrange
            md_file = self.obsidian_publish_dir / "empty_frontmatter.md"
            md_file.write_text("""---
---
# Content""")
            
            # Act
            result = read_md_metadata(md_file)
            
            # Assert
            self.assertEqual(result, {})
            
            # Cleanup
            md_file.unlink()

        # Test 5: Values with colons
        with self.subTest("Values with colons"):
            # Arrange
            md_file = self.obsidian_publish_dir / "colons_in_values.md"
            md_file.write_text("""---
Title: Document: With Colon
Time: 12:30:45
---
# Content""")
            
            # Act
            result = read_md_metadata(md_file)
            
            # Assert
            self.assertEqual(result, {
                'Title': 'Document: With Colon',
                'Time': '12:30:45'
            })
            
            # Cleanup
            md_file.unlink()

        # Test 6: Invalid lines in front matter
        with self.subTest("Invalid lines in front matter"):
            # Arrange
            md_file = self.obsidian_publish_dir / "invalid_lines.md"
            md_file.write_text("""---
Title: Invalid Lines
This line has no colon
Author: Smith
---
# Content""")
            
            # Act
            result = read_md_metadata(md_file)
            
            # Assert
            self.assertEqual(result, {
                'Title': 'Invalid Lines',
                'Author': 'Smith'
            })
            
            # Cleanup
            md_file.unlink()
    def test_parse_date(self):
        """Test parsing dates in various formats"""
        test_cases = [
            # (input_date, expected_day, expected_month, expected_year)
            ("2024-12-20", 20, 12, 2024),
            ("20.12.2024", 20, 12, 2024),
            ("20/12/2024", 20, 12, 2024),
            ("12/20/2024", 20, 12, 2024),
            ("December 20, 2024", 20, 12, 2024),
            ("20 December 2024", 20, 12, 2024),
            ("20241220", 20, 12, 2024),
            # Not in supported formats, should fail
            ("2024.12.20", None, None, None),
            ("invalid-date", None, None, None),
            ("", None, None, None),
            ("20-12-2024", None, None, None),  
            ("15 Jan 2023", None, None, None),
        ]

        for date_str, expected_day, expected_month, expected_year in test_cases:
            with self.subTest(date_str=date_str):
                result = parse_date(date_str)
                if expected_day is None:
                    self.assertIsNone(result)
                else:
                    self.assertEqual(result.day, expected_day, date_str)
                    self.assertEqual(result.month, expected_month, date_str)
                    self.assertEqual(result.year, expected_year, date_str)


    def test_slugify(self):
        """Test filename slugification for Jekyll"""
        # Test 1: Basic filename conversion
        with self.subTest("Basic filename conversion"):
            test_file = self.obsidian_publish_dir / "Hello World!.md"
            test_file.touch()
            self.assertEqual(slugify(test_file), "hello-world.md")
            test_file.unlink()

        # Test 2: Filename with special characters
        with self.subTest("Filename with special characters"):
            test_file = self.obsidian_publish_dir / "File@Name#123.pdf"
            test_file.touch()
            self.assertEqual(slugify(test_file), "filename123.pdf")
            test_file.unlink()

        # Test 3: Jekyll post with valid date
        with self.subTest("Jekyll post with valid date"):
            test_file = self.obsidian_publish_dir / "Post.md"
            test_file.write_text("""---
layout: post
date: 2024-12-20
---
# Content""")
            self.assertEqual(slugify(test_file), "2024-12-20-post.md")
            test_file.unlink()

        # Test 4: Jekyll post with missing date
        with self.subTest("Jekyll post with missing date"):
            test_file = self.obsidian_publish_dir / "Bad Post.md"
            test_file.write_text("""---
layout: post
---
# Content""")
            with self.assertRaises(PublishTransformError) as cm:
                slugify(test_file)
            self.assertIn("Missing date field", str(cm.exception))
            test_file.unlink()

        # Test 5: Jekyll post with invalid date
        with self.subTest("Jekyll post with invalid date"):
            test_file = self.obsidian_publish_dir / "Bad Post.md"
            test_file.write_text("""---
layout: post
date: invalid-date
---
# Content""")
            with self.assertRaises(PublishTransformError):
                slugify(test_file)
            test_file.unlink()

        # Test 6: Date formatting with leading zeros
        with self.subTest("Date formatting with leading zeros"):
            test_file = self.obsidian_publish_dir / "Post.md"
            test_file.write_text("""---
layout: post
date: 2024-1-5
---
# Content""")
            self.assertEqual(slugify(test_file), "2024-01-05-post.md")
            test_file.unlink()

        # Test 7: Unicode handling
        with self.subTest("Unicode handling"):
            test_file = self.obsidian_publish_dir / "Café_Über.md"
            test_file.touch()
            self.assertEqual(slugify(test_file), "cafe-uber.md")
            test_file.unlink()

        # Test 8: Multiple spaces/special chars
        with self.subTest("Multiple spaces/special chars"):
            test_file = self.obsidian_publish_dir / "  File   Name--with___chars!.txt"
            test_file.touch()
            self.assertEqual(slugify(test_file), "file-name-with-chars.txt")
            test_file.unlink()

    def test_transform_md_ref(self):
        """Unit test for transform_md_ref"""
        test_cases = [
            ("Reference", ""),
            ("SMT#bit-vectors", ""),
            ("RealReference | Text", "Text"),
            ("RealReference|Text", "Text"),
            ("RealReference |Text", "Text"),
            ("Image.png|200", ""),  # Numeric only
            ("Image.png|Alt Text", "Alt Text"),
            ("Image.png|Alt Text|200", "Alt Text"),
            ("", "")
        ]
        
        for input_ref, expected_output in test_cases:
            with self.subTest(input_ref=input_ref):
                self.assertEqual(transform_md_ref(input_ref), expected_output, input_ref)


    def test_transform_image_ref(self):
        """
        Unit test for transform_image_ref
        Test cases:
            1. ![[image.png]]                   → ![Image](image.png)
            2. ![[image.png|200]]               → ![Image](image.png){:width="200"}
            3. ![[image.png|200x100]]           → ![Image](image.png){:width="200" height="100"}
            4. ![[image.png|alt text]]          → ![alt text](image.png)
            5. ![[image.png|alt text|200]]      → ![alt text](image.png){:width="200"}
            6. ![[subdir/image.png]]            → ![Image](subdir/image.png)
            7. ![[image.png|alt text|200x100]]  → ![alt text](image.png){:width="200" height="100"}
        """
        with patch('obsidian_to_jekyll.JEKYLL_ROOT', self.jekyll_root), \
            patch('obsidian_to_jekyll.JEKYLL_IMAGE_DIR', self.jekyll_img_dir):

            relative_path = self.jekyll_img_dir.relative_to(self.jekyll_root)

            test_cases = [
                ("image.png", 
                (f"![Image]({relative_path}/image.png)", "image.png")),
                ("image.png|200", 
                (f'![Image]({relative_path}/image.png){{:width="200"}}', "image.png")),
                ("image.png|200x100", 
                (f'![Image]({relative_path}/image.png){{:width="200" height="100"}}', "image.png")),
                ("subdir/image.png|Alt Text", 
                (f"![Alt Text]({relative_path}/subdir/image.png)", "subdir/image.png")),
                ("image.png|Alt Text|200x100", 
                (f'![Alt Text]({relative_path}/image.png){{:width="200" height="100"}}', "image.png"))
            ]
            
            for input_ref, (expected_output, expected_path) in test_cases:
                with self.subTest(input_ref=input_ref):
                    result_output, result_path = transform_image_ref(input_ref)
                    self.assertEqual(result_output, expected_output)
                    self.assertEqual(result_path, Path(expected_path))

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    @patch('obsidian_to_jekyll.copy_file')
    def test_ensure_image_available(self, mock_copy, mock_mkdir, mock_exists):
        """Unit test for ensure_image_available with mocked filesystem"""
        with self.subTest("Image exists"):
            mock_exists.return_value = True
            src = self.fake_image
            dst = self.jekyll_img_dir / "fake_tmp" /self.fake_image.name
            
            # Test successful copy
            self.assertTrue(ensure_image_available(src, dst))
            mock_mkdir.assert_not_called()
            mock_copy.assert_called_once_with(src, dst)
            
            # Test missing source
            mock_exists.return_value = False
            with self.assertRaises(PublishTransformError):
                ensure_image_available(src, dst)

        with self.subTest("Image doesn't exist"):
            mock_exists.return_value = False

            src = self.obsidian_img_dir / "missing.png"
            dst = self.jekyll_img_dir / "missing.png"
        
            with self.assertRaises(PublishTransformError):
                ensure_image_available(src, dst)

    # def test_transform_md_match(self):
    #     image_test = ['!', 'alt text', 'subdir/image.png', '![alt text | JEKYLL_IMG_DIR/image.png]']
    #     external_link_test = ['', '']
    #     # External links []()
    #     match_mock = Mock()
    #     match_mock.group.side_effect = 

    #     result = transform_md_match(match_mock)
    #     print(f"RESULT: {result}")

if __name__ == "__main__":
    unittest.main()