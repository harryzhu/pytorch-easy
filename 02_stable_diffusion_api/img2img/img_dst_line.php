<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title></title>
	<style type="text/css">
		table{
			border-collapse: collapse;
		}
		td{
			border: 3px solid green;
			max-width: 130px;
			min-width: 130px;
			font-size: 13px;
			text-align: center;
			color: #ccc;
		}

		tr{
			border: 1px solid #ff0000;
			padding: 10px 0;
		}

		img {
			width: 128px;
			height: auto;
			border-radius: 5px;
		}

		a.btn-disable{
			color: #ccc;
		}
	</style>
	
	<script type="text/javascript" 	src="asset/js/jquery-3.7.1.min.js"></script>
</head>
<body>
	<?php
	$images = array();
	$files = scandir("data/img_dst_line");
	foreach ($files as $file) {
		if(substr($file, -4)!=".png"){
			continue;
		}
		array_push($images, $file);
	}

	sort($images);
	echo "<table>";
	$row_size = 30;
	$image_group = array_chunk($images, $row_size);
	$rows = sizeof($image_group);
	for ($i=0;$i<$rows;$i++) {
		echo "<tr>";

		foreach ($image_group[$i] as $image) {
			if(trim($image)==""){
				continue;
			}
			echo '<td>';
			$sub_dir = "data/img_dst_line/".$image;
			$sub_images = scandir($sub_dir);

			foreach ($sub_images as $simg) {
				if(substr($simg, -4)!=".png"){
					continue;
				}
				if(sizeof($sub_images) > 2 && strpos($simg, "_".(sizeof($sub_images)-3).".png") > 0){
					continue;
				}
				$btn_del = "<button href='' class='btn-normal' data-src='".$sub_dir."'>Delete</button>";
				echo '<img src="'.$sub_dir.'/'.$simg.'?v='.time().'" title="'.$simg.'"/>'.$btn_del;
			}

			
			echo '</td>';
		}

		echo "</tr>";

		
	}

	echo "</table>";


	?>

	<script type="text/javascript">
		counter = 1;
		$("button").click(function(){
			var tdir = $(this).data("src")
			console.log(tdir)
			$.ajax({
				url:"delete.php?d="+tdir,
				success:function(result){
					//alert(result);
					if(result == "ok"){
						if(counter % 10 == 0){
							window.location.reload()
							counter = 1;
						}
						counter += 1
						console.log(result)
					}
				}});
		});  


	</script>

</body>
</html>